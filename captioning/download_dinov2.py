"""Download pinned Meta pure-vision DINOv2 weights, without remote code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).parent / '.weights' / 'dinov2-small'

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AIAA2205-research'}),timeout=60)

if __name__ == '__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with fetch('https://huggingface.co/api/models/facebook/dinov2-small') as r:
        meta=json.load(r)
    revision=meta['sha']
    files={x['rfilename'] for x in meta['siblings']}
    weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    manifest={'repository':'facebook/dinov2-small','revision':revision,'files':[]}
    for name in ['config.json','preprocessor_config.json',weight]:
        url=f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}'
        target=ROOT/name
        with fetch(url) as r, target.with_suffix('.download').open('wb') as out:
            while chunk:=r.read(1024*1024): out.write(chunk)
        target.with_suffix('.download').replace(target)
        item={'file':name,'url':url,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        manifest['files'].append(item)
        print(json.dumps(item),flush=True)
    (ROOT/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
