import torch
import torch.nn as nn

__all__ = ['cosine_sim_loss', 'complex_cosine_sim_loss']

def cosine_sim_loss(real_z, noise_z):
    cos = nn.CosineSimilarity(dim=-2, eps=1e-6)
    distance = cos(real_z, noise_z)
    return 1-torch.mean(distance)

def complex_cosine_sim_loss(real_freq_z, noise_freq_z):
    real_freq_z = torch.stack([real_freq_z.real, real_freq_z.imag], dim=-1)
    noise_freq_z = torch.stack([noise_freq_z.real, noise_freq_z.imag], dim=-1)
    cos = nn.CosineSimilarity(dim=-3, eps=1e-6)
    distance = cos(real_freq_z, noise_freq_z)
    return 1-torch.mean(distance)