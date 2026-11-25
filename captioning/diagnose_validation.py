"""Validation-only diagnostics; outputs never become training targets."""
import csv
import json
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from train import tokenize
from transformers import AutoModel, AutoTokenizer


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    run = root / 'captioning/runs/resnet50_aoa_v1'
    labels = list(csv.DictReader((root/'Validation_Labels.csv').open(encoding='utf-8-sig')))
    preds = list(csv.DictReader((run/'validation_predictions_reloaded.csv').open(encoding='utf-8-sig')))
    lookup = {p['image_id']: p['description'] for p in preds}
    ids = [Path(r['image_filename']).stem for r in labels]
    assert len(lookup) == len(preds) == len(labels) == 2500 and set(ids) == set(lookup)
    references = [r['description'] for r in labels]
    predictions = [lookup[i] for i in ids]
    model_dir = root/'Text_Embedding_Model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    embedder = AutoModel.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    def encode(texts):
        vectors=[]
        for start in range(0,len(texts),64):
            tokens=tokenizer(texts[start:start+64],padding=True,truncation=True,max_length=64,return_tensors='pt')
            tokens={k:v.cuda() for k,v in tokens.items()}
            hidden=embedder(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
            vectors.append(F.normalize(pooled.float(),dim=1).cpu())
        return torch.cat(vectors)
    scores=(encode(references)*encode(predictions)).sum(1).tolist()
    records=[]
    for row, reference, prediction, score in zip(labels,references,predictions,scores):
        words=tokenize(prediction)
        trigrams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
        with Image.open(root/'Validation'/row['image_filename']) as image:
            w,h=image.size
        # Approximation ignoring integer rounding in resize(256).
        retained=(224/256)**2*min(w,h)/max(w,h)
        records.append(dict(image_id=Path(row['image_filename']).stem,reference=reference,
            prediction=prediction,cosine=score,reference_words=len(tokenize(reference)),
            prediction_words=len(words),repeated_trigram=len(set(trigrams))<len(trigrams),
            at_length_cap=len(words)>=24,center_crop_retained_area=retained))
    def group(predicate):
        subset=[r for r in records if predicate(r)]
        return dict(count=len(subset),mean_cosine=sum(r['cosine'] for r in subset)/len(subset) if subset else None)
    summary=dict(mean_cosine=sum(scores)/len(scores),
        reference_mean_words=sum(r['reference_words'] for r in records)/len(records),
        prediction_mean_words=sum(r['prediction_words'] for r in records)/len(records),
        repeated_trigram=group(lambda r:r['repeated_trigram']),
        length_cap=group(lambda r:r['at_length_cap']),
        score_below_04=group(lambda r:r['cosine']<.4),
        score_above_07=group(lambda r:r['cosine']>=.7),
        long_reference=group(lambda r:r['reference_words']>=15),
        other_reference=group(lambda r:r['reference_words']<15),
        mean_crop_retained_area=sum(r['center_crop_retained_area'] for r in records)/len(records),
        crop_retains_less_than_half=group(lambda r:r['center_crop_retained_area']<.5),
        unique_predictions=len(set(predictions)))
    (run/'diagnostics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (run/'validation_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2),flush=True)
    print('Lowest-scoring validation pairs:')
    for r in sorted(records,key=lambda r:r['cosine'])[:10]:
        print(json.dumps(r,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
