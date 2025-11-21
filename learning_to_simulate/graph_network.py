import torch
import torch.nn as nn
from typing import Callable, Optional, List

def build_mlp(
    input_size: int, 
    hidden_sizes: List[int], 
    output_size: Optional[int]=None,
    activation:nn.Module=nn.ReLU(),
    activate_final: bool=False,
    layer_norm: bool=False
) -> nn.Module:
    sizes = [input_size] + hidden_sizes
    
    if output_size is not None:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i],sizes[[i+1]]))