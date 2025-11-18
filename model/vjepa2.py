import torch

# preprocessor
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
# models
vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')    