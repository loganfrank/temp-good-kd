import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as v2F

# Define the normalization statistics of each dataset
CIFAR100 = {'mean': [0.4802, 0.4481, 0.3975], 'std': [0.2764, 0.2688, 0.2815]}
IMAGENET = {'mean': [0.4850, 0.4560, 0.4060], 'std': [0.2290, 0.2240, 0.2250]}
PETS = {'mean': [0.4810, 0.4452, 0.3950], 'std': [0.2603, 0.2547, 0.2620]}
HALF = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

dataset_statistics = {
    'cifar100': CIFAR100,
    'imagenet': IMAGENET,
    'pets': PETS,
    'half': HALF,
}

#############################################
##### GETS ARBITRARY DATA AUGMENTATIONS #####
#############################################

def get_transforms(operations, normalization, image_size, device='cpu', **kwargs):

    # Key-word arguments
    randaug = kwargs.get('randaug', 'rand-n2-m14')
    randaug_n = int(re.search(r"n(\d+)-m(\d+)", randaug).group(1))
    randaug_m = int(re.search(r"n(\d+)-m(\d+)", randaug).group(2))
    hflip_p = kwargs.get('hflip_p', 0.5)
    padding = kwargs.get('padding', 12)
    color_jitter_prob = kwargs.get('colorjitter_prob', None)
    color_jitter = kwargs.get('colorjitter_values', [0.4, 0.4, 0.4])
    
    # Retrieve the means of the desired normalization
    mean = dataset_statistics[normalization]['mean']
    mean_int = [round(255 * v) for v in mean]
    std = dataset_statistics[normalization]['std']

    # List for holding the list of augmentations
    batch_augmentations = []
    augmentations = []

    ######################
    ##### PREPROCESS #####
    ######################

    # PIL -> Tensor
    if 'piltensor' in operations:
        batch_augmentations.append(v2.PILToTensor())
    
    # NumPy -> Tensor
    if 'nptensor' in operations:
        batch_augmentations.append(NumPyToTensor())

    # Resize
    if 'resize' in operations:
        batch_augmentations.append(v2.Resize(image_size))

    # Center Image Cropping (mainly for test examples only)
    if 'centercrop' in operations:
        batch_augmentations.append(v2.CenterCrop(image_size))

    # Random Image Cropping (mainly for train examples only)
    if 'randomcrop' in operations:
        batch_augmentations.append(v2.RandomCrop(image_size, padding=None))

    ##############################
    ##### DATA AUGMENTATIONS #####
    ##############################

    # RandAugment through timm
    if 'randaug' in operations:
        augmentations.append(v2.RandAugment(num_ops=randaug_n, magnitude=randaug_m, fill=mean_int))

    # Color jittering
    if 'colorjitter' in operations:
        augmentations.append(v2.RandomApply([v2.ColorJitter(*color_jitter)], p=color_jitter_prob) if color_jitter_prob is not None else v2.ColorJitter(*color_jitter))

    # Horizontal flipping
    if 'hflip' in operations:
        augmentations.append(v2.RandomHorizontalFlip(p=hflip_p))

    # Padding image with mean then randomly crop again
    if 'padcrop' in operations:
        augmentations.append(v2.RandomCrop(image_size, padding=padding, fill=mean_int))

    ##########################
    ##### POSTPROCESSING #####
    ##########################

    # Finally convert to float32 at the end (should be fastest here)
    augmentations.append(v2.ToDtype(torch.float32, scale=True))

    # Normalization
    augmentations.append(v2.Normalize(mean=mean, std=std))

    #########################################################
    ##### SEQUENTIAL OBJECT FOR PROCESSING BACK-TO-BACK #####
    #########################################################

    # Convert all of them to nn.Sequential objects
    batch_augmentations = nn.Sequential(*batch_augmentations) if len(batch_augmentations) > 0 else None
    augmentations = nn.Sequential(*augmentations) if len(augmentations) > 0 else None

    return batch_augmentations, augmentations

###################################
##### MIXUP DATA AUGMENTATION #####
###################################

class Mixup(nn.Module):
    def __init__(self, alpha, device='cpu'):
        super().__init__()

        # The alpha value for mixing up images, lower will give more weight to original input
        self.alpha = alpha

        # Keep track of the device
        self.device = device

        # Beta distribution is the normal distribution for sampling in mixup
        c0 = torch.tensor(self.alpha[0], device=self.device)
        c1 = torch.tensor(self.alpha[1], device=self.device)

        # NOTE: PyTorch is weird and has concentration1 as the alpha and concentration0 as the beta: x ~ Beta(alpha, beta)
        # but I want c0 to be alpha and b1 to be beta
        self._dist = torch.distributions.Beta(concentration1=c0, concentration0=c1)

    def forward(self, x, labels):
        # Get the number of examples in the batch
        batch_size = len(x)

        # Do a random permutation of the batch
        random_indexes = torch.randperm(batch_size)
        
        # Get the uniformly distributed alphas
        alphas = self._dist.sample((batch_size,))

        # Apply mixup
        x = (torch.ones_like(alphas) - alphas)[:, None, None, None] * x + (alphas)[:, None, None, None] * x[random_indexes]
        labels = (torch.ones_like(alphas) - alphas)[:, None] * labels + (alphas)[:, None] * labels[random_indexes]

        return x, labels

#########################################################
##### TRANSFORM FOR CONVERTING FROM NUMPY TO TENSOR #####
#########################################################

class NumPyToTensor(nn.Module):
    def forward(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)