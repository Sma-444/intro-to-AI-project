"""Re-load a checkpoint and validate on all official validation pairs, never Test/."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train import AttentionCaptioner, CaptionDataset, collate_train, evaluate, get_transforms, read_labels
from model_io import model_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(8)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ck["config"]
    # Transformer trials used benchmark=True during checkpoint selection.
    # Match it on reload: FP16 beam decisions can change with cuDNN kernels.
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark',
        config['architecture'].endswith('_spatial_transformer'))
    model = model_from_checkpoint(ck)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = read_labels(root / "Validation_Labels.csv")
    assert len(rows) == 2500
    _, transform = get_transforms(config.get("preprocessing", "resize256_centercrop224"),
                                  config.get("image_size", 224))
    dataset = CaptionDataset(rows, root / "Validation", ck["vocab"], transform, config["max_caption_length"])
    loader = DataLoader(dataset, batch_size=48, num_workers=0, shuffle=False, collate_fn=collate_train)
    score, output = evaluate(model, loader, ck["vocab"], rows, root, device, device.type == "cuda", 24,
                             checkpoint_path.parent / "validation_predictions_reloaded.csv",
                             decode_policy=config.get("decode_policy", "greedy"),
                             length_alpha=config.get("length_alpha", 0.6))
    import csv
    with output.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["image_id", "description"]
        predictions = list(reader)
    assert len(predictions) == len(rows)
    assert [p["image_id"] for p in predictions] == [Path(r["image_filename"]).stem for r in rows]
    assert len({p["image_id"] for p in predictions}) == 2500
    assert all(p["description"].strip() for p in predictions)
    assert checkpoint_path.stat().st_size < 200_000_000
    result = dict(epoch=ck["epoch"], architecture=config["architecture"],
        decoder_layers=config.get('decoder_layers'),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        decode_policy=config.get("decode_policy", "greedy"),
        preprocessing=config.get("preprocessing", "resize256_centercrop224"),
        image_size=config.get("image_size", 224),
        checkpoint_bytes=checkpoint_path.stat().st_size, validation_rows=len(rows),
        score_on_reload=score, score_recorded=ck["validation_similarity"],
        baseline_resnet18_score=0.528876, difference_vs_baseline=score-0.528876,
        baseline_resnet50_score=0.5779770612716675,
        difference_vs_resnet50=score-0.5779770612716675,
        baseline_aoa_score=0.5824428796768188, difference_vs_aoa=score-0.5824428796768188,
        best_aoa_beam_score=0.5869023203849792, difference_vs_aoa_beam=score-0.5869023203849792)
    (checkpoint_path.parent / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
