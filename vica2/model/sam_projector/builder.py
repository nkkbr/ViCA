import torch.nn as nn

def build_sam_projector():
    
    # sam2_hidden_size = 1152 # only for sam2.1-hiera-large
    sam2_hidden_size = 896 # only for sam2.1-hiera-base_plus
    hidden_size = 3584 # only for Qwen-2-7B
    
    modules = [nn.Linear(sam2_hidden_size, hidden_size)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_size, hidden_size))

    return nn.Sequential(*modules)