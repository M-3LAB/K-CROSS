import torch
import torch.nn as nn

__all__ = ['cosine_similarity', 'l1_diff', 'l2_diff', 'freq_distance']

def cosine_similiarity(real_z, fake_z):
    """
    Cosine Similiarity 

    Args:
        real_z (vector): the hidden space of real image
        fake_z (vector): the hidden space of fake image 
    
    Output:
        cosine similiarity distance between two hidden space 
    
    """
    cos = nn.CosineSimilarity(dim=-2, eps=1e-6)
    distance = cos(real_z, fake_z)
    return torch.mean(distance)

def l1_diff(real_z, fake_z):
    """
    L1 Difference

    Args:
        real_z (vector): the hidden space of real image
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l1 difference between two hidden space 
    
    """
    diff_tensor = real_z - fake_z
    distance = torch.norm(diff_tensor, p=1, dim=-2, keepdim=True)
    return torch.mean(distance)

def l2_diff(real_z, fake_z):
    """
    L2 Difference

    Args:
        real_z (vector): the hidden space of real image 
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l2 difference between two hidden space
    """
    diff_tensor = real_z - fake_z
    distance = torch.norm(diff_tensor, p=2, dim=-2, keepdim=True)
    return torch.mean(distance)

def freq_distance(real_z, fake_z):
    real_z_freq = torch.stack([real_z.real, real_z.imag], dim=-1)
    fake_z_freq = torch.stack([fake_z.real, fake_z.imag], dim=-1)

    # frequency distance using (squared) Euclidean distance
    tmp = (real_z_freq - fake_z_freq) ** 2
    freq_distance = torch.sqrt(tmp[..., 0] + tmp[..., 1])

    return torch.mean(freq_distance)
