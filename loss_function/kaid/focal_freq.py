import torch
import torch.nn as nn

__all__ = ['FocalFreqLoss']

class FocalFreqLoss(nn.Module):
    def __init__(self, loss_weight, patch_factor=1, alpha=1.0, log_matrix=False, 
                 avg_spectrum=False, batch_matrix=False):
        super(FocalFreqLoss, self).__init__()

        self.loss_weight = loss_weight
        self.patch_factor = patch_factor
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.avg_spectrum = avg_spectrum
        self.batch_matrix = batch_matrix

    def loss_formulation(self, real_freq, recon_freq, 
                         assigned_weight_matrix=None):

        if assigned_weight_matrix is not None:
            weight_matrix = assigned_weight_matrix.detach() 
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp.real + matrix_tmp.imag) * self.alpha


    def forward(self, x):
        pass