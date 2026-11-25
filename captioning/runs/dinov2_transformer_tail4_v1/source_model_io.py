"""Construct generation models from versioned or legacy checkpoint metadata."""
from train import AttentionCaptioner


def model_from_checkpoint(checkpoint):
    config = checkpoint['config']
    if config['architecture'] in ('resnet50_spatial_transformer', 'dinov2_small_spatial_transformer'):
        from transformer_model import TransformerCaptioner
        fields = ['d_model', 'nhead', 'visual_layers', 'decoder_layers',
                  'dim_feedforward', 'dropout', 'max_caption_length']
        model = TransformerCaptioner(len(checkpoint['vocab']), pretrained=False,
                                     backbone_name=config.get('backbone_name', 'resnet50'),
                                     vision_config=config.get('vision_config'),
                                     **{key: config[key] for key in fields})
        model.config = dict(config)
        model.config.setdefault('backbone_name', 'resnet50')
    else:
        model = AttentionCaptioner(len(checkpoint['vocab']),
            embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'],
            dropout=config['dropout'], pretrained=False,
            backbone_name=config.get('backbone_name', 'resnet18'),
            attention_type=config.get('attention_type', 'additive'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model
