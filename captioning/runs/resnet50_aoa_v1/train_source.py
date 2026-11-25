import argparse
import csv
import math
import os
import json
import shutil
import random
import re
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_labels(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_filename", "description"]:
            raise ValueError(f"Unexpected label columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows or any(not row["description"].strip() for row in rows):
        raise ValueError(f"Empty labels or descriptions in {path}")
    return rows


def build_vocab(rows, min_frequency=2):
    counts = {}
    for row in rows:
        for token in tokenize(row["description"]):
            counts[token] = counts.get(token, 0) + 1
    words = sorted(word for word, count in counts.items() if count >= min_frequency)
    return {word: index for index, word in enumerate(SPECIALS + words)}


def encode_caption(text, vocab, max_length):
    ids = [BOS]
    ids.extend(vocab.get(token, UNK) for token in tokenize(text))
    ids.append(EOS)
    if len(ids) > max_length:
        ids = ids[:max_length]
        ids[-1] = EOS
    return ids


class CaptionDataset(Dataset):
    def __init__(self, rows, image_dir, vocab, transform, max_length):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / row["image_filename"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        caption = encode_caption(row["description"], self.vocab, self.max_length)
        return image, torch.tensor(caption, dtype=torch.long), row["description"]


def collate_train(batch):
    images, captions, references = zip(*batch)
    max_length = max(len(caption) for caption in captions)
    padded = torch.full((len(captions), max_length), PAD, dtype=torch.long)
    for index, caption in enumerate(captions):
        padded[index, :len(caption)] = caption
    return torch.stack(images), padded, references


class AttentionCaptioner(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, dropout=0.3,
                 pretrained=True, backbone_name="resnet18", attention_type="additive"):
        super().__init__()
        self.backbone_name = backbone_name
        self.attention_type = attention_type
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.visual_projection = nn.Identity()
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.visual_projection = nn.Linear(2048, 512)
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            self.visual_projection = nn.Linear(768, 512)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.encoder = backbone.features if backbone_name == "convnext_tiny" else nn.Sequential(*list(backbone.children())[:-2])
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.feature_attention = nn.Linear(512, hidden_dim, bias=False)
        self.state_attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.initial_h = nn.Linear(512, hidden_dim)
        self.initial_c = nn.Linear(512, hidden_dim)
        self.cell = nn.LSTMCell(embedding_dim + 512, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + 512, vocab_size)
        self.hidden_dim = hidden_dim
        if attention_type == "aoa":
            self.aoa_information = nn.Linear(hidden_dim + 512, 512)
            self.aoa_gate = nn.Linear(hidden_dim + 512, 512)
        elif attention_type != "additive":
            raise ValueError(f"Unknown attention type: {attention_type}")

    def encode(self, images):
        spatial = self.visual_projection(self.encoder(images).flatten(2).transpose(1, 2))
        return spatial, self.feature_attention(spatial)

    def attend(self, spatial, projected, hidden):
        scores = self.score_attention(torch.tanh(projected + self.state_attention(hidden).unsqueeze(1)))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), spatial).squeeze(1)
        return context

    def start_state(self, spatial):
        pooled = spatial.mean(dim=1)
        return torch.tanh(self.initial_h(pooled)), torch.tanh(self.initial_c(pooled))

    def step(self, token, spatial, projected, state):
        hidden, cell = state
        context = self.attend(spatial, projected, hidden)
        if self.attention_type == "aoa":
            query_context = torch.cat([hidden, context], dim=1)
            context = self.aoa_information(query_context) * torch.sigmoid(self.aoa_gate(query_context))
        embedded = self.embedding(token)
        hidden, cell = self.cell(torch.cat([self.dropout(embedded), context], dim=1), (hidden, cell))
        logits = self.output(torch.cat([self.dropout(hidden), self.dropout(context)], dim=1))
        return logits, (hidden, cell)

    def forward(self, images, captions):
        spatial, projected = self.encode(images)
        state = self.start_state(spatial)
        outputs = []
        for position in range(captions.size(1) - 1):
            logits, state = self.step(captions[:, position], spatial, projected, state)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    @torch.inference_mode()
    def generate(self, images, max_words=24):
        spatial, projected = self.encode(images)
        state = self.start_state(spatial)
        token = torch.full((images.size(0),), BOS, dtype=torch.long, device=images.device)
        finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        predictions = []
        for _ in range(max_words):
            logits, state = self.step(token, spatial, projected, state)
            token = logits.argmax(dim=1)
            token = torch.where(finished, torch.full_like(token, PAD), token)
            predictions.append(token)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return torch.stack(predictions, dim=1)


def decode_ids(ids, inverse_vocab):
    words = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS:
            break
        if token_id >= len(SPECIALS):
            words.append(inverse_vocab[token_id])
    return " ".join(words)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomApply([transforms.ColorJitter(0.12, 0.12, 0.12, 0.03)], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    return train_transform, eval_transform


def evaluate(model, loader, vocab, val_rows, data_root, device, amp, max_words,
             prediction_path=None):
    model.eval()
    inverse_vocab = [None] * len(vocab)
    for word, index in vocab.items():
        inverse_vocab[index] = word
    predictions = []
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                generated = model.generate(images, max_words=max_words)
            predictions.extend(decode_ids(row, inverse_vocab) for row in generated.cpu())

    prediction_path = Path(prediction_path) if prediction_path else data_root / "captioning" / "validation_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "description"])
        for row, description in zip(val_rows, predictions):
            writer.writerow([Path(row["image_filename"]).stem, description])

    return mean_embedding_similarity(
        [row["description"] for row in val_rows], predictions,
        data_root / "Text_Embedding_Model", device,
    ), prediction_path


def mean_embedding_similarity(references, predictions, model_dir, device, batch_size=64):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    embedder = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()

    def encode(texts):
        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=64, return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                hidden = embedder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
                results.append(F.normalize(pooled.float(), p=2, dim=1))
        return torch.cat(results)

    reference_vectors = encode(references)
    prediction_vectors = encode(predictions)
    if reference_vectors.shape[1] != 384:
        raise ValueError(f"Expected 384-dimensional text embeddings, got {reference_vectors.shape}")
    score = (reference_vectors * prediction_vectors).sum(dim=1).mean().item()
    del embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def save_checkpoint(path, model, vocab, epoch, score, args):
    payload = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embedding_dim": 256,
            "hidden_dim": 512,
            "dropout": 0.3,
            "max_caption_length": args.max_caption_length,
            "architecture": f"{model.backbone_name}_spatial_{model.attention_type}_attention_lstm",
            "backbone_name": model.backbone_name,
            "attention_type": model.attention_type,
            "preprocessing": "resize256_centercrop224",
        },
        "epoch": epoch,
        "validation_similarity": score,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    if temporary.stat().st_size >= 200_000_000:
        raise ValueError("Checkpoint exceeds the course 200 MB size limit")
    temporary.replace(path)


def train(args):
    data_root = Path(args.data_root).resolve()
    train_rows = read_labels(data_root / "Training_Labels.csv")
    val_rows = read_labels(data_root / "Validation_Labels.csv")
    vocab = build_vocab(train_rows, min_frequency=args.min_frequency)
    train_transform, eval_transform = get_transforms()
    train_set = CaptionDataset(
        train_rows, data_root / "Training", vocab, train_transform, args.max_caption_length,
    )
    val_set = CaptionDataset(
        val_rows, data_root / "Validation", vocab, eval_transform, args.max_caption_length,
    )
    loader_options = {"num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    if args.workers:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_train, **loader_options,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_train, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.cpu_threads)
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = AttentionCaptioner(len(vocab), backbone_name=args.backbone, attention_type=args.attention).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": args.decoder_lr},
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.csv"
    if history_path.exists() or (run_dir / "best_model.pt").exists():
        raise FileExistsError(f"Choose a new output directory; run already exists: {run_dir}")
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if not history_path.exists():
        with history_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["epoch", "token_loss", "validation_similarity", "minutes"])
    best_score = -math.inf
    stale_epochs = 0

    print(f"Device: {device}; images: {len(train_set)} train / {len(val_set)} validation")
    print(f"Backbone: {args.backbone}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary: {len(vocab)}; batch size: {args.batch_size}; checkpoint dir: {run_dir}")
    print(f"Attention: {args.attention}; encoder frozen for {args.freeze_epochs} epochs, then fine-tune final stage")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.encoder[-1].parameters():
                parameter.requires_grad = True
            print("Unfroze final encoder stage for low-rate fine-tuning")

        model.train()
        model.encoder.eval()
        if epoch > args.freeze_epochs:
            model.encoder[-1].train()
        started = time.time()
        loss_sum = 0.0
        token_count = 0
        for step, (images, captions, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            valid_tokens = targets.ne(PAD).sum().item()
            loss_sum += loss.item() * valid_tokens
            token_count += valid_tokens
            if step % args.log_every == 0 or step == len(train_loader):
                print(f"epoch {epoch:02d}/{args.epochs} step {step:04d}/{len(train_loader)} "
                      f"token_loss={loss_sum / max(token_count, 1):.4f}", flush=True)

        val_score, prediction_path = evaluate(
            model, val_loader, vocab, val_rows, data_root, device, amp, args.max_words,
            run_dir / "validation_predictions.csv",
        )
        epoch_loss = loss_sum / max(token_count, 1)
        elapsed = time.time() - started
        print(f"epoch {epoch:02d}: loss={epoch_loss:.4f}, validation_similarity={val_score:.6f}, "
              f"time={elapsed / 60:.1f} min, predictions={prediction_path}", flush=True)
        with history_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{epoch_loss:.6f}", f"{val_score:.6f}",
                                    f"{elapsed / 60:.3f}"])
        save_checkpoint(run_dir / "last_model.pt", model, vocab, epoch, val_score, args)
        if val_score > best_score:
            best_score = val_score
            stale_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, vocab, epoch, val_score, args)
            shutil.copyfile(prediction_path, run_dir / "validation_predictions_best.csv")
            size_mb = (run_dir / "best_model.pt").stat().st_size / 1e6
            print(f"Saved new best checkpoint: {best_score:.6f} ({size_mb:.1f} MB)", flush=True)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale validation epochs", flush=True)
            break
        if epoch > args.freeze_epochs:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-attention-LSTM image captioner.")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "convnext_tiny"], default="resnet18")
    parser.add_argument("--attention", choices=["additive", "aoa"], default="additive")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "runs" / "resnet18_attn_lstm"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-caption-length", type=int, default=34)
    parser.add_argument("--max-words", type=int, default=24)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--lr-decay", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2205)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args


if __name__ == "__main__":
    train(parse_args())
