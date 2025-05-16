import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(logits, eps=1e-6):
    mean = logits.mean(dim=-1, keepdims=True)
    stddev = logits.std(dim=-1, keepdims=True)
    return (logits - mean) / (stddev + eps)

class KLDivergence():
    def __init__(self, temperature=1, logit_normalization=False):
        self.temperature = temperature # int: >1 flattens, <1 peaks
        self.inverse_temperature = 1 / temperature # float <1 flattens, >1 peaks

        # Flag for normalizing the logits before softmax following CVPR24 paper
        self.logit_normalization = logit_normalization
        
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def __repr__(self):
        return f'KLDivergence (temperature={self.temperature}, logit_normalization={self.logit_normalization})'

    def __call__(self, p, q):
        """
        In knowledge distillation,
        p: teacher logits
        q: student logits

        Note: PyTorch KL goes student then teacher for its arguments
        """

        # Apply optional logit normalization
        if self.logit_normalization:
            p = normalize(p)
            q = normalize(q)

        return (self.temperature ** 2) * self.kl(F.log_softmax(q * self.inverse_temperature, dim=1), F.softmax(p * self.inverse_temperature, dim=1))

class SoftCrossEntropy():
    def __init__(self, temperature=1, logit_normalization=False):
        self.temperature = temperature # int: >1 flattens, <1 peaks
        self.inverse_temperature = 1 / temperature # float <1 flattens, >1 peaks

        # Flag for normalizing the logits before softmax following CVPR24 paper
        self.logit_normalization = logit_normalization

    def __call__(self, p, q, p_is_logits=False, q_is_logits=True):
        """
        Will operate whether categorical or soft cross entropy
        p: teacher logits
        q: student logits
        """

        # Transform p if it is logits
        if p_is_logits:
            # Apply optional logit normalization
            if self.logit_normalization:
                p = normalize(p)

            p = F.softmax(p, dim=1)

        # Transform (and log) q if it is logits, otherwise just log
        if q_is_logits:
            # Apply optional logit normalization
            if self.logit_normalization:
                q = normalize(q)

            q = F.log_softmax(q, dim=1)
            # return torch.mean(torch.sum(-p * F.log_softmax(q, dim=1), dim=1))
        else:
            q = F.log(q, dim=1)
            # return torch.mean(torch.sum(-p * F.log(q, dim=1), dim=1))

        return torch.mean(-torch.sum(p * q, dim=1))

class Entropy():
    def __init__(self, temperature=10, num_classes=-1, reduce_mean=True):
        self.temperature = temperature # int: >1 flattens, <1 peaks
        self.inverse_temperature = 1 / temperature # float <1 flattens, >1 peaks
        self.reduce_mean = reduce_mean

        if num_classes > 0:
            self.max_entropy = torch.log(torch.tensor(num_classes))
        else:
            self.max_entropy = None

    def __call__(self, p, p_is_logits=True):
        """
        Will operate whether categorical or soft cross entropy
        p: teacher logits
        q: student logits
        """
        # If logits were provided, softmax them to form a distribution
        if p_is_logits:
            p = F.softmax(p, dim=1)
            
        # Compute entropy
        e = -torch.sum(p * torch.log(p), dim=1)

        # Reduce the entropy across examples into a mean (for a batch loss)
        if self.reduce_mean:
            e = torch.mean(e)

        # Make it relative to the max entropy possible (uniform)
        if self.max_entropy is not None:
            e = e / self.max_entropy
        
        return e