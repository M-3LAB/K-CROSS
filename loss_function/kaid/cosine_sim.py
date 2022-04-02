import torch
import torch.nn as nn

__all__ = ['cosine_sim_loss']

def cosine_sim_loss(real_z, noise_z):
    cos = nn.CosineSimilarity(dim=-2, eps=1e-6)
    distance = cos(real_z, noise_z)
    return torch.mean(distance)