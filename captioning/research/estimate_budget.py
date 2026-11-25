"""Count a proposed architecture's tensors on CPU; no training or weight downloads."""
from pathlib import Path
import json
import torch
from torch import nn
from torchvision.models import resnet50

torch.manual_seed(2205)
backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
d = 256
parts = nn.ModuleDict({
    'encoder': backbone,
    'projection': nn.Linear(2048, d),
    'visual_norm': nn.LayerNorm(d),
    'visual_relation': nn.TransformerEncoderLayer(d, 4, 1024, batch_first=True, norm_first=True),
    'decoder': nn.ModuleList([nn.TransformerDecoderLayer(d, 4, 1024, batch_first=True, norm_first=True) for _ in range(2)]),
    'embedding': nn.Embedding(1417, d),
    'text_position': nn.Embedding(32, d),
    'output': nn.Linear(d, 1417),
    'final_norm': nn.LayerNorm(d),
})
result = {
    'description': 'ResNet50, 256 dimensions, 1 visual relation layer, 2 decoder layers, 4 heads, FFN1024; fixed 2D spatial positions assumed',
    'parameters_by_component': {k: sum(p.numel() for p in v.parameters()) for k, v in parts.items()},
    'total_parameters': sum(p.numel() for p in parts.parameters()),
    'state_dict_tensor_bytes': sum(t.numel() * t.element_size() for t in parts.state_dict().values()),
    'notes': 'CPU tensor count only. Serialized metadata overhead, training VRAM, speed and accuracy are not measured. No end-to-end captioner implemented here.',
}
(Path(__file__).parent / 'proposed_budget.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
