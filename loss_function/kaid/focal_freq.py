import torch
import torch.nn as nn
#from model.kaid.complex_nn.fourier_transform import *

__all__ = ['FocalFreqLoss']

class FocalFreqLoss(nn.Module):
    def __init__(self, loss_weight, alpha=1.0, log_matrix=False, 
                 avg_spectrum=False, batch_matrix=False):
        super(FocalFreqLoss, self).__init__()
        """The torch.nn.Module class that implements focal frequency loss - a
            frequency domain loss function for optimizing generative models.

            Ref:
            Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
            <https://arxiv.org/pdf/2012.12821.pdf>

            Args:
                loss_weight (float): weight for focal frequency loss. Default: 1.0
                alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
                patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
                ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
                log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
                batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
        """

        self.loss_weight = loss_weight
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.avg_spectrum = avg_spectrum
        self.batch_matrix = batch_matrix
    
    def freq_stack(self, freq):
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq

    def loss_formulation(self, real_freq, recon_freq, 
                         assigned_weight_matrix=None):
        # spectrum weight matrix
        if assigned_weight_matrix is not None:
            # if the matrix is predefined
            weight_matrix = assigned_weight_matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred_freq, target_freq, assigned_weight_matrix=None):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            assigned_weight_matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.freq_stack(pred_freq)
        target_freq = self.freq_stack(target_freq)
        #print(f'pred_freq.size(): {pred_freq.size()}')
        #print(f'target_freq.size(): {target_freq.size()}')

        # whether to use minibatch average spectrum
        if self.avg_spectrum:
            pred_freq = torch.mean(pred_freq, dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)
        
        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, assigned_weight_matrix) * self.loss_weight