"""Read local lecture slide text with stdlib; never execute slide content."""
from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET
import json

root = Path(__file__).resolve().parents[3]
output = Path(__file__).parent / 'course_text'
output.mkdir(parents=True, exist_ok=True)
index = []
for folder in [root/'课件', root/'L02']:
    for path in sorted(folder.rglob('*.pptx')):
        pages = []
        with ZipFile(path) as archive:
            names = sorted((n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)),
                           key=lambda n:int(re.search(r'slide(\d+)',n).group(1)))
            for name in names:
                tree = ET.fromstring(archive.read(name))
                paragraphs = [' '.join(t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'))
                              for p in tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}p')]
                number = int(re.search(r'slide(\d+)',name).group(1))
                pages.append({'slide':number,'text':'\n'.join(p for p in paragraphs if p)})
        target = output/(path.stem+'.txt')
        target.write_text('\n\n'.join(f"SLIDE {p['slide']}\n{p['text']}" for p in pages),encoding='utf-8')
        index.append({'path':str(path.relative_to(root)),'slides':len(pages),'extracted_text':str(target.relative_to(output.parent))})
(output/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(index,ensure_ascii=True,indent=2))
