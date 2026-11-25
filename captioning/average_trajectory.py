"""Average two checkpoints from one training trajectory, without label fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from model_io import model_from_checkpoint


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = args.run_dir.resolve()
    out = args.output_dir.resolve()
    paths = [source/'best_model.pt', source/'last_model.pt']
    states = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    first, last = states
    if first['config'] != last['config'] or first['vocab'] != last['vocab']:
        raise ValueError('Configuration/vocabulary mismatch')
    if first['epoch'] >= last['epoch']:
        raise ValueError('Need distinct best and later final checkpoints')
    a, b = first['state_dict'], last['state_dict']
    if a.keys() != b.keys():
        raise ValueError('Tensor key mismatch')
    weights = {}
    for key in a:
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f'Tensor mismatch: {key}')
        if x.is_floating_point():
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f'Nonfinite source: {key}')
            weights[key] = (x.double().mul(.5) + y.double().mul(.5)).to(x.dtype)
        else:
            if not torch.equal(x, y):
                raise ValueError(f'Nonfloating state differs: {key}')
            weights[key] = x.clone()
    provenance = dict(method='equal_average_same_trajectory_best_and_last',
        sources=[dict(path=str(p), sha256=digest(p), epoch=s['epoch'], weight=.5)
                 for p, s in zip(paths, states)],
        training_args=json.loads((source/'config.json').read_text(encoding='utf-8'))['args'])
    checkpoint = dict(state_dict=weights, config=first['config'], vocab=first['vocab'],
        epoch=None, validation_similarity=None, weight_averaging=provenance)
    model_from_checkpoint(checkpoint)  # Strict construction validates all tensor keys.
    out.mkdir(parents=True, exist_ok=True)
    target = out/'averaged_model.pt'
    if target.exists():
        raise FileExistsError(target)
    temporary = out/'averaged_model.tmp'
    torch.save(checkpoint, temporary)
    if temporary.stat().st_size >= 200_000_000:
        raise ValueError('Inference checkpoint exceeds 200 MB')
    restored = torch.load(temporary, map_location='cpu', weights_only=True)
    assert all(torch.equal(weights[k], restored['state_dict'][k]) for k in weights)
    temporary.replace(target)
    provenance.update(checkpoint_bytes=target.stat().st_size, sha256=digest(target))
    (out/'averaging.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
