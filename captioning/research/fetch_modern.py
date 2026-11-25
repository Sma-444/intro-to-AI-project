"""Read official modern VLM project descriptions; never download model weights."""
import concurrent.futures
import datetime
import json
import sys
from pathlib import Path
from fetch_sources import repo

if __name__ == '__main__':
    specs = [
        ('salesforce/LAVIS', ['README.md', 'projects/blip2/README.md']),
        ('QwenLM/Qwen3-VL', ['README.md']),
        ('microsoft/Florence-2', ['README.md']),
        ('huggingface/smollm', ['vision/README.md']),
    ]
    vision_only = '--vision-only' in sys.argv
    if vision_only:
        specs = [('facebookresearch/dinov2', ['README.md']),
                 ('facebookresearch/dinov3', ['README.md'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(repo, specs))
    target = Path(__file__).parent / ('modern_visual_sources.json' if vision_only else 'modern_sources.json')
    target.write_text(json.dumps({'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'repositories': results}, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
