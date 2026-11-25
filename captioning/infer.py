import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train import decode_ids, get_transforms
from decoding_search import search_features
from model_io import model_from_checkpoint


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.paths = sorted(Path(image_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return path.stem, tensor


def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--decode-policy", choices=["greedy", "constrained_greedy", "beam3"], default=None)
    parser.add_argument("--length-alpha", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark',
        config['architecture'].endswith('_spatial_transformer'))
    args.max_words = args.max_words if args.max_words is not None else config.get("max_words", 24)
    args.decode_policy = args.decode_policy or config.get("decode_policy", "greedy")
    args.length_alpha = args.length_alpha if args.length_alpha is not None else config.get("length_alpha", 0.6)
    model = model_from_checkpoint(checkpoint)
    model.to(device).eval()
    _, eval_transform = get_transforms(config.get("preprocessing", "resize256_centercrop224"),
                                      config.get("image_size", 224))
    dataset = ImageDataset(args.image_dir, eval_transform)
    if not dataset:
        raise ValueError(f"No JPG images found under {args.image_dir}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    inverse_vocab = [None] * len(vocab)
    for word, index in vocab.items():
        inverse_vocab[index] = word

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "description"])
        with torch.inference_mode():
            for image_ids, images in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    if args.decode_policy == "greedy":
                        generated = model.generate(images, max_words=args.max_words)
                    else:
                        spatial, projected = model.encode(images)
                        generated = search_features(model, spatial, projected,
                            beam_size=3 if args.decode_policy == "beam3" else 1,
                            alpha=args.length_alpha if args.decode_policy == "beam3" else 0.,
                            constrain=True, max_words=args.max_words)
                for image_id, tokens in zip(image_ids, generated.cpu()):
                    description = decode_ids(tokens, inverse_vocab)
                    if not description:
                        raise ValueError(f"Model generated an empty description for {image_id}")
                    writer.writerow([image_id, description])
                    rows_written += 1

    print(f"Wrote {rows_written} predictions to {output_path}")


if __name__ == "__main__":
    main()
