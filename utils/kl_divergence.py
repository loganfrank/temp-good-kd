import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(logits, eps=1e-6):
    mean = logits.mean(dim=-1, keepdims=True)
    stddev = logits.std(dim=-1, keepdims=True)
    return (logits - mean) / (stddev + eps)

class KLDivergence():
    def __init__(self, temperature=1, logit_normalization=False, **kwargs):

        # Temperature scaling for softening distributions
        self.temperature = temperature # int: >1 flattens, <1 peaks
        self.inverse_temperature = 1 / temperature # float <1 flattens, >1 peaks

        # Flag for normalizing the logits before softmax following CVPR24 paper
        self.logit_normalization = logit_normalization
        
        # Actual function
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def __repr__(self):
        return f'KLDivergence (temperature={self.temperature}, logit_normalization={self.logit_normalization})'

    def __call__(self, p, q, **kwargs):
        """
        p: teacher logits
        q: student logits
        """

        # Apply optional logit normalization
        if self.logit_normalization:
            p = normalize(p)
            q = normalize(q)

        # Loss expects (input, target) but KL(P||Q) is KL(target||input)
        return (self.temperature ** 2) * self.kl(F.log_softmax(q * self.inverse_temperature, dim=1), F.softmax(p * self.inverse_temperature, dim=1))