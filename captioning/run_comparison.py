"""Run the two approved architecture experiments sequentially on one GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    base = Path(__file__).resolve().parent
    root = base.parent
    os.environ['TORCH_HOME'] = str(base / '.weights')
    experiments = [('resnet50_aoa_v1', 'resnet50', 'aoa'),
                   ('convnext_tiny_attn_v1', 'convnext_tiny', 'additive')]
    results = []
    for name, backbone, attention in experiments:
        run = base / 'runs' / name
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'training_history.csv').exists():
            raise FileExistsError(f'Run already exists: {run}')
        shutil.copy2(base / 'train.py', run / 'train_source.py')
        command = [sys.executable, '-u', str(base/'train.py'), '--backbone', backbone,
                   '--attention', attention, '--output-dir', str(run), '--batch-size', '24',
                   '--eval-batch-size', '48', '--workers', '2', '--cpu-threads', '8', '--epochs', '16']
        print(f'START {name}', flush=True)
        with (run/'console.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'TRAINING FINISHED {name}; validating saved checkpoint', flush=True)
        with (run/'verification.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(base/'evaluate_saved.py'), '--checkpoint',
                            str(run/'best_model.pt')], cwd=root, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads((run/'verification.json').read_text(encoding='utf-8'))
        result['experiment'] = name
        results.append(result)
        (base/'runs'/'architecture_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(result), flush=True)
    print('COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
