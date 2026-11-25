"""Fixed stratified visual audit of validation predictions, never Test/."""
import csv
import json
import random
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

root=Path(__file__).resolve().parents[1]
folder=root/'captioning/runs/dinov2_transformer_v1'
with (folder/'paired_validation.csv').open(encoding='utf-8',newline='') as f:
    rows=list(csv.DictReader(f))
for row in rows:
    for old,new in [('dino_cosine','candidate_cosine'),('dino_prediction','candidate_prediction'),('resnet_prediction','control_prediction')]:
        if old not in row:
            row[old]=row[new]
rng=random.Random(2205)
groups=[('low',lambda s:s<.4,8),('middle',lambda s:.4<=s<.7,2),('high',lambda s:s>=.7,2)]
selected=[]
for name,predicate,count in groups:
    for row in rng.sample([r for r in rows if predicate(float(r['dino_cosine']))],count):
        selected.append(dict(row,stratum=name))
out=root/'captioning/research/validation_visual_audit'
out.mkdir(exist_ok=True)
(out/'selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for page in range(3):
    sheet=Image.new('RGB',(1120,1440),'white')
    draw=ImageDraw.Draw(sheet)
    for j,row in enumerate(selected[page*4:page*4+4]):
        top=j*360
        with Image.open(root/'Validation'/(row['image_id']+'.jpg')) as im:
            im=im.convert('RGB')
            original=ImageOps.contain(im,(350,245))
            sheet.paste(original,(8,top+30))
            w,h=im.size
            size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
            resized=im.resize(size,Image.Resampling.BILINEAR)
            left=round((size[0]-224)/2);upper=round((size[1]-224)/2)
            crop=resized.crop((left,upper,left+224,upper+224))
            sheet.paste(crop,(375,top+30))
        draw.text((8,top+4),f"{row['image_id']} | {row['stratum']} | cosine={float(row['dino_cosine']):.4f}",font=font,fill='black')
        draw.text((8,top+280),'Original image',font=small,fill='black')
        draw.text((375,top+280),'Actual center crop',font=small,fill='black')
        y=top+32
        for name,key in [('Reference','reference'),('DINO prediction','dino_prediction'),('ResNet prediction','resnet_prediction')]:
            for line in textwrap.wrap(name+': '+row[key],width=53):
                draw.text((620,y),line,font=font,fill='black')
                y+=23
            y+=12
        draw.line((0,top+355,1120,top+355),fill='#aaaaaa')
    sheet.save(out/f'page_{page+1}.jpg',quality=93)
print(str(out))
