"""Small spatial Transformer captioner; only the ImageNet CNN is pretrained."""
import math
import json
from pathlib import Path
import torch
from torch import nn
from torchvision import models

PAD, BOS, EOS = 0, 1, 2


class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, visual_layers=1,
                 decoder_layers=2, dim_feedforward=1024, dropout=0.2,
                 max_caption_length=34, pretrained=True, backbone_name='resnet50',
                 vision_config=None):
        super().__init__()
        self.config = dict(architecture=f'{backbone_name}_spatial_transformer', version=1,
            backbone_name=backbone_name,
            d_model=d_model, nhead=nhead, visual_layers=visual_layers,
            decoder_layers=decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, max_caption_length=max_caption_length,
            preprocessing='resize256_centercrop224', image_size=224,
            decode_policy='beam3', length_alpha=0.6, max_words=24)
        if backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            channels = 2048
        elif backbone_name == 'dinov2_small':
            from transformers import Dinov2Config, Dinov2Model
            folder = Path(__file__).parent / '.weights' / 'dinov2-small'
            if vision_config is None:
                vision_config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
            cfg = Dinov2Config(**vision_config)
            cfg._attn_implementation = 'sdpa'
            if pretrained:
                self.encoder, info = Dinov2Model.from_pretrained(folder, config=cfg,
                    local_files_only=True, output_loading_info=True)
                if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys'):
                    raise RuntimeError(f'Unexpected DINOv2 load: {info}')
                self.config['vision_provenance'] = json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
            else:
                self.encoder = Dinov2Model(cfg)
            self.config['vision_config'] = vision_config
            channels = cfg.hidden_size
        else:
            raise ValueError(f'Unsupported visual encoder: {backbone_name}')
        self.projection = nn.Linear(channels, d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.visual_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True,
            norm_first=True) for _ in range(visual_layers)])
        self.visual_output_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True,
            norm_first=True), decoder_layers, norm=nn.LayerNorm(d_model))
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_caption_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)
        # Initialize cloned attention layers independently, preserving ImageNet weights.
        for name, parameter in self.named_parameters():
            if not name.startswith('encoder.') and parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def spatial_positions(self, height, width, device, dtype):
        d = self.config['d_model']
        if d % 4:
            raise ValueError('2D sinusoidal positions require d_model divisible by four')
        frequency = torch.exp(torch.arange(d // 4, device=device).float()
                              * (-math.log(10000.0) / (d // 4)))
        y, x = torch.meshgrid(torch.arange(height, device=device),
                              torch.arange(width, device=device), indexing='ij')
        x = x.flatten().float()[:, None] * frequency
        y = y.flatten().float()[:, None] * frequency
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1).to(dtype)

    def encode(self, images):
        if self.config['backbone_name'] == 'resnet50':
            features = self.encoder(images)
            tokens = features.flatten(2).transpose(1, 2)
            height, width = features.shape[2:]
        else:
            # Remove CLS; keep the 16x16 spatial patch tokens at 224 input.
            tokens = self.encoder(pixel_values=images).last_hidden_state[:, 1:]
            patch = self.encoder.config.patch_size
            height, width = images.shape[2]//patch, images.shape[3]//patch
            if tokens.shape[1] != height*width:
                raise ValueError('Unexpected DINO patch token count')
        memory = self.visual_norm(self.projection(tokens))
        memory = self.dropout(memory + self.spatial_positions(
            height, width, memory.device, memory.dtype))
        for layer in self.visual_layers:
            memory = layer(memory)
        memory = self.visual_output_norm(memory)
        # Second element preserves the shared beam-search interface.
        return memory, memory.new_empty((memory.shape[0], 0))

    def unfreeze_visual_tail(self):
        if self.config['backbone_name'] == 'resnet50':
            modules = [self.encoder[-1]]
        else:
            modules = [*self.encoder.encoder.layer[-2:], self.encoder.layernorm]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def decode(self, tokens, memory):
        length = tokens.shape[1]
        if length > self.position.num_embeddings:
            raise ValueError('Caption prefix exceeds configured position embeddings')
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) * math.sqrt(self.config['d_model'])
        hidden = self.dropout(hidden + self.position(positions))
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        hidden = self.decoder(hidden, memory, tgt_mask=causal,
                              tgt_key_padding_mask=tokens.eq(PAD))
        return self.output(hidden)

    def forward(self, images, captions):
        memory, _ = self.encode(images)
        return self.decode(captions[:, :-1], memory)

    def start_state(self, spatial):
        return (torch.empty(spatial.shape[0], 0, dtype=torch.long, device=spatial.device),)

    def step(self, token, spatial, projected, state):
        prefix = torch.cat((state[0], token[:, None]), dim=1)
        return self.decode(prefix, spatial)[:, -1], (prefix,)

    @torch.inference_mode()
    def generate(self, images, max_words=24):
        from decoding_search import search_features
        spatial, projected = self.encode(images)
        return search_features(self, spatial, projected, beam_size=1,
                               alpha=0, constrain=False, max_words=max_words)
