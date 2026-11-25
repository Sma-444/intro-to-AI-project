"""Download public research text for inspection, never execute downloaded code."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import urllib.request

OUT = Path(__file__).parent / 'github_sources'
OUT.mkdir(exist_ok=True)

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'AIAA2205-course-research', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=35) as response:
        return response.read()

def repo(spec):
    name, files = spec
    result = {'repo': name, 'files': []}
    try:
        meta = json.loads(get('https://api.github.com/repos/' + name))
        commit = json.loads(get('https://api.github.com/repos/' + name + '/commits/' + meta['default_branch']))['sha']
        result.update(commit=commit, license=meta.get('license'), url=meta['html_url'])
        folder = OUT / name.replace('/', '__')
        folder.mkdir(exist_ok=True)
        for file in files:
            url = f'https://raw.githubusercontent.com/{name}/{commit}/{file}'
            item = {'path': file, 'url': url, 'permalink': f'https://github.com/{name}/blob/{commit}/{file}'}
            try:
                data = get(url)
                target = folder / file
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            except Exception as exc:
                item['error'] = str(exc)
            result['files'].append(item)
    except Exception as exc:
        result['error'] = str(exc)
    return result

if __name__ == '__main__':
    query = 'image+captioning+flickr8k+transformer+language%3APython'
    evidence = {'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'search_url': 'https://api.github.com/search/repositories?q=' + query + '&per_page=8'}
    try:
        search = json.loads(get(evidence['search_url']))
        evidence['search_results'] = [{k: r[k] for k in ['full_name', 'html_url', 'description', 'stargazers_count']} for r in search['items']]
    except Exception as exc:
        evidence['search_error'] = str(exc)
    specs = [
        ('sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning', ['README.md', 'models.py', 'train.py']),
        ('husthuaan/AoANet', ['README.md', 'models/AoAModel.py', 'opts.py']),
        ('ruotianluo/ImageCaptioning.pytorch', ['README.md', 'captioning/models/AttModel.py', 'captioning/models/TransformerModel.py', 'tools/train.py', 'configs/transformer.yml']),
        ('aimagelab/meshed-memory-transformer', ['README.md', 'models/transformer/encoders.py', 'models/transformer/decoders.py', 'train.py']),
        ('keras-team/keras-io', ['examples/vision/image_captioning.py']),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        evidence['repositories'] = list(pool.map(repo, specs))
    (OUT / 'manifest.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=True, indent=2))
