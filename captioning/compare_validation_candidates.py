"""Paired validation-only diagnostics. Embeddings never enter generation or training."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from train import read_labels


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-dir',type=Path,default=root/'captioning/runs/dinov2_transformer_v1')
    parser.add_argument('--control-dir',type=Path,default=root/'captioning/runs/resnet50_transformer_v1')
    args=parser.parse_args()
    out=args.candidate_dir.resolve()
    labels=read_labels(root/'Validation_Labels.csv')
    ids=[Path(row['image_filename']).stem for row in labels]
    def predictions(path):
        with path.open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==2500 and len({r['image_id'] for r in rows})==2500
        values={r['image_id']:r['description'] for r in rows}
        assert set(values)==set(ids)
        return [values[key] for key in ids]
    control=predictions(args.control_dir/'validation_predictions_reloaded.csv')
    candidate=predictions(out/'validation_predictions_reloaded.csv')
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(root/'Text_Embedding_Model',local_files_only=True)
    model=AutoModel.from_pretrained(root/'Text_Embedding_Model',local_files_only=True).cuda().eval()
    @torch.inference_mode()
    def embed(texts):
        values=[]
        for i in range(0,len(texts),64):
            tokens=tokenizer(texts[i:i+64],padding=True,truncation=True,max_length=64,return_tensors='pt').to('cuda')
            hidden=model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
            values.append(F.normalize(pooled.float(),dim=1).cpu())
        return torch.cat(values)
    refs=embed([r['description'] for r in labels])
    before=(refs*embed(control)).sum(1).numpy()
    after=(refs*embed(candidate)).sum(1).numpy()
    delta=after-before
    rng=np.random.default_rng(2205)
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
    with (out/'paired_validation.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['image_id','reference','control_prediction','candidate_prediction','control_cosine','candidate_cosine','difference'])
        for i,key in enumerate(ids):
            writer.writerow([key,labels[i]['description'],control[i],candidate[i],float(before[i]),float(after[i]),float(delta[i])])
    result=dict(rows=2500,control_dir=str(args.control_dir.resolve()),candidate_dir=str(out),
        control_mean=float(before.mean()),candidate_mean=float(after.mean()),
        mean_difference=float(delta.mean()),improved=int((delta>0).sum()),worse=int((delta<0).sum()),
        equal=int((delta==0).sum()),control_below_04=int((before<.4).sum()),candidate_below_04=int((after<.4).sum()),
        control_above_07=int((before>=.7).sum()),candidate_above_07=int((after>=.7).sum()),
        paired_bootstrap_95_percent_interval=np.quantile(means,[.025,.975]).tolist(),
        caveat='Conditional on this reused validation set and these selected checkpoints. Bootstrap does not measure training-seed variance or remove model-selection bias.')
    (out/'paired_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
