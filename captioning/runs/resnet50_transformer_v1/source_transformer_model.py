"""Small spatial Transformer captioner; only the ImageNet CNN is pretrained."""
import math
import torch
from torch import nn
from torchvision import models

PAD, BOS, EOS = 0, 1, 2


class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, visual_layers=1,
                 decoder_layers=2, dim_feedforward=1024, dropout=0.2,
                 max_caption_length=34, pretrained=True):
        super().__init__()
        self.config = dict(architecture='resnet50_spatial_transformer', version=1,
            d_model=d_model, nhead=nhead, visual_layers=visual_layers,
            decoder_layers=decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, max_caption_length=max_caption_length,
            preprocessing='resize256_centercrop224', image_size=224,
            decode_policy='beam3', length_alpha=0.6, max_words=24)
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Linear(2048, d_model)
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
        features = self.encoder(images)
        memory = self.visual_norm(self.projection(features.flatten(2).transpose(1, 2)))
        memory = self.dropout(memory + self.spatial_positions(
            features.shape[2], features.shape[3], memory.device, memory.dtype))
        for layer in self.visual_layers:
            memory = layer(memory)
        memory = self.visual_output_norm(memory)
        # Second element preserves the shared beam-search interface.
        return memory, memory.new_empty((memory.shape[0], 0))

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
