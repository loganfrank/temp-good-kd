import torch
import torch.nn as nn

import math

KAIMING_MODELS = [
    'resnet18',
]
TRUNC_NORMAL_MODELS = [
]

def init_network(network, name):
    # Vision Transformers or ConvNeXts
    if name in TRUNC_NORMAL_MODELS:
        trunc_normal(network, std=0.02, negative_bound=-2, positive_bound=2)
    
    # ResNets, MobileNet, etc.
    elif name in KAIMING_MODELS:
        kaiming(network, name, fan='in')

    # Set biases to 0
    zero_biases(network)

# Kaiming initialization which is the default for ResNets
def kaiming(network, name, fan='in', activation=nn.ReLU):
    mode = 'fan_in' if fan == 'in' else 'fan_out'
    nonlinearity = 'relu' if activation == nn.ReLU else 'leaky_relu'

    for n, m in network.named_modules():
        # Convolutional layers
        if isinstance(m, (nn.Conv2d)):
            # Kaiming normal initialization
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, a=(0 if nonlinearity == 'relu' else 0.01))
        
        # Linear layers
        elif isinstance(m, nn.Linear):
            if 'mobilenet' in name:
                nn.init.normal_(m.weight, 0, 0.01)
            else:
                # Kaiming uniform initialization
                nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity, a=math.sqrt(5))

# Trunc normal initialization which is default for Vision Transformers and ConvNeXts (a lot of larger models)
def trunc_normal(network, std=0.02, negative_bound=-2, positive_bound=2):
    for n, m in network.named_modules():
        # Transformation layers: conv, linear, etc.
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=std, a=negative_bound, b=positive_bound)

# Set the biases in the network to 0
def zero_biases(network):
    for m in network.modules():
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)