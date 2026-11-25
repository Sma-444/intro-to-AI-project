"""Predefined validation-only decoding comparison on the fixed best AoA checkpoint."""
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train import (AttentionCaptioner, CaptionDataset, collate_train, decode_ids,
                   get_transforms, read_labels, mean_embedding_similarity)
from decoding_search import search_features


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'captioning/runs/decoding_aoa_v1'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise FileExistsError(output)
    ckpath = root/'captioning/runs/resnet50_aoa_v1/best_model.pt'
    ck = torch.load(ckpath, map_location='cpu', weights_only=True)
    torch.set_num_threads(8)
    model = AttentionCaptioner(len(ck['vocab']), pretrained=False,
        backbone_name='resnet50', attention_type='aoa').cuda().eval()
    model.load_state_dict(ck['state_dict'], strict=True)
    rows = read_labels(root/'Validation_Labels.csv')
    assert len(rows) == 2500
    _, transform = get_transforms()
    data = CaptionDataset(rows, root/'Validation', ck['vocab'], transform, 34)
    loader = DataLoader(data, batch_size=48, collate_fn=collate_train, num_workers=0)
    vocab = [None]*len(ck['vocab'])
    for word, index in ck['vocab'].items():
        vocab[index] = word
    variants = {
        'original_greedy': dict(beam_size=1, alpha=0., constrain=False),
        'constrained_greedy': dict(beam_size=1, alpha=0., constrain=True),
        'beam3_alpha0': dict(beam_size=3, alpha=0., constrain=True),
        'beam3_alpha06': dict(beam_size=3, alpha=0.6, constrain=True),
    }
    (output/'config.json').write_text(json.dumps(dict(checkpoint=str(ckpath),
        max_words=24, variants=variants),indent=2),encoding='utf-8')
    predictions = {name:[] for name in variants}
    with torch.inference_mode():
        for batch, (images, _, _) in enumerate(loader):
            images = images.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                spatial, projected = model.encode(images)
                for name, settings in variants.items():
                    tokens = search_features(model, spatial, projected, **settings)
                    text = [decode_ids(x, vocab) for x in tokens.cpu()]
                    predictions[name].extend(text)
                    if batch == 0 and name == 'original_greedy':
                        original = [decode_ids(x, vocab) for x in model.generate(images).cpu()]
                        assert text == original, 'Beam width 1 must reproduce original greedy text'
            if (batch+1)%10 == 0:
                print(f'Validation images processed: {len(predictions["original_greedy"])}/2500',flush=True)
    results=[]
    for name, texts in predictions.items():
        assert len(texts)==2500 and all(t.strip() for t in texts)
        with (output/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image_id','description'])
            writer.writerows((Path(row['image_filename']).stem,t) for row,t in zip(rows,texts))
        score=mean_embedding_similarity([r['description'] for r in rows], texts,
                                       root/'Text_Embedding_Model',torch.device('cuda'))
        repeats=sum(len(set(tuple(t.split()[i:i+3]) for i in range(len(t.split())-2)))<max(0,len(t.split())-2) for t in texts)
        result=dict(variant=name,score=score,mean_words=sum(len(t.split()) for t in texts)/2500,
                    repeated_trigram_count=repeats,**variants[name])
        results.append(result)
        print(json.dumps(result),flush=True)
        (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('COMPLETE: fixed-weight decoding comparison',flush=True)


if __name__=='__main__':
    main()
