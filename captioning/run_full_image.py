"""Full-frame 224 experiment; test 320 only if 224 beats the crop control by >0.002."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(size):
    base = Path(__file__).resolve().parent
    root = base.parent
    run_dir = base/'runs'/f'resnet50_aoa_letterbox{size}_v1'
    if (run_dir/'training_history.csv').exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for source in ('train.py', 'evaluate_saved.py', 'infer.py'):
        shutil.copy2(base/source, run_dir/f'{Path(source).stem}_source.py')
    env = dict(python=sys.version, packages={n:metadata.version(n) for n in
        ('torch','torchvision','transformers','numpy','Pillow','safetensors')})
    (run_dir/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    os.environ['TORCH_HOME'] = str(base/'.weights')
    command = [sys.executable, '-u', str(base/'train.py'), '--backbone','resnet50',
        '--attention','aoa','--preprocessing','letterbox','--image-size',str(size),
        '--output-dir',str(run_dir),'--batch-size','24','--eval-batch-size','48',
        '--workers','2','--cpu-threads','8','--epochs','16']
    print(f'START full-image {size}', flush=True)
    with (run_dir/'console.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'TRAINING FINISHED {size}; verifying checkpoint',flush=True)
    with (run_dir/'verification.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',str(base/'evaluate_saved.py'),'--checkpoint',
            str(run_dir/'best_model.pt')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = json.loads((run_dir/'verification.json').read_text(encoding='utf-8'))
    print(json.dumps(result),flush=True)
    return result


def main():
    base=Path(__file__).resolve().parent
    results=[run(224)]
    path=base/'runs/full_image_comparison.json'
    path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    if results[0]['difference_vs_aoa'] > .002:
        print('224 passed predeclared +0.002 threshold; checking 320 resource usage',flush=True)
        subprocess.run([sys.executable,'-u',str(base/'check_upgrade.py'),'--backbone','resnet50',
            '--attention','aoa','--preprocessing','letterbox','--image-size','320'],check=True)
        results.append(run(320))
        path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    else:
        print('224 did not pass +0.002 threshold; no 320 run',flush=True)
    print('FULL IMAGE COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    main()
