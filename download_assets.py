from pathlib import Path
import argparse, hashlib, json, urllib.request, zipfile
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--all',action='store_true',help='Also restore optional history and course material')
args=parser.parse_args()
for a in json.loads((root/'assets-manifest.json').read_text())['assets']:
    if a['group'].startswith('optional-') and not args.all: continue
    dest=root/'.downloads'/a['name'];dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_suffix('.part')
        print('Downloading',a['name'],flush=True)
        urllib.request.urlretrieve(a['url'],tmp);tmp.replace(dest)
    with dest.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=a['sha256']:raise ValueError('Checksum mismatch: '+a['name'])
    with zipfile.ZipFile(dest) as z:
        for member in z.infolist():
            if not (root/member.filename).resolve().is_relative_to(root):raise ValueError('Unsafe ZIP path')
        z.extractall(root)
    print('Verified and restored',a['name'],flush=True)
