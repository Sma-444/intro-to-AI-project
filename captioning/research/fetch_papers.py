"""Fetch paper abstracts and public repository support files, without execution."""
from pathlib import Path
from html.parser import HTMLParser
import concurrent.futures
import json
from fetch_sources import get, repo

OUT = Path(__file__).parent / 'paper_sources'
OUT.mkdir(exist_ok=True)

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())

def paper(identifier):
    url = 'https://arxiv.org/abs/' + identifier
    result = {'id': identifier, 'url': url}
    try:
        data = get(url).decode('utf-8')
        parser = Text()
        parser.feed(data)
        (OUT / (identifier + '.txt')).write_text('\n'.join(parser.parts), encoding='utf-8')
        result['status'] = 'abstract page downloaded, not full paper'
    except Exception as exc:
        result['error'] = str(exc)
    return result

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(paper, ['1502.03044', '1908.06954', '1912.08226', '2001.03615']))
    extra = repo(('husthuaan/AoANet', ['data/README.md', 'train.sh']))
    manifest = {'papers': results, 'additional_repository_files': extra}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
