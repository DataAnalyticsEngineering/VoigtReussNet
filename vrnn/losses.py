import torch
from vrnn.transformations import OrthogonalTransformation
import torch.nn as nn
from vrnn.tensortools import unpack_sym

class VoigtReussNormalizedLoss(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.dim = dim
        self.orth = OrthogonalTransformation(dim)
        self.onebydim_sqrt = 1 / dim ** 0.5

    def forward(self, pred, truth):
        eigs_pred = pred[..., :self.dim]
        q_scaled_pred = pred[..., self.dim:]
        eigs_truth = truth[..., :self.dim]
        q_scaled_truth = truth[..., self.dim:]

        # Batch transform q_scaled back to Q matrices
        Q_pred = self.orth.inverse_transform(q_scaled_pred)
        Q_truth = self.orth.inverse_transform(q_scaled_truth)

        QEpred = torch.einsum('...ij,...j->...ij', Q_pred, eigs_pred)
        QEtruth = torch.einsum('...ij,...j->...ij', Q_truth, eigs_truth)
        
        kappa_pred = torch.einsum('...ij,...kj->...ik', QEpred, Q_pred)
        kappa_truth = torch.einsum('...ij,...kj->...ik', QEtruth, Q_truth)

        diff = kappa_pred - kappa_truth
        frob_norm_diff = torch.sqrt(torch.sum(diff * diff, dim=(-2, -1)))
        
        return torch.mean(frob_norm_diff) * self.onebydim_sqrt
    
class RelativeFrobeniusLoss(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.dim = dim
    
    def forward(self, pred, truth):
        
        kappa_pred = unpack_sym(pred, dim=self.dim)
        kappa_truth = unpack_sym(truth, dim=self.dim)

        diff = kappa_pred - kappa_truth
        frob_norm_diff = torch.sqrt(torch.sum(diff * diff, dim=(-2, -1)))
        frob_norm_truth = torch.sqrt(torch.sum(kappa_truth * kappa_truth, dim=(-2, -1)))
        relative_error = frob_norm_diff / frob_norm_truth
        
        return torch.mean(relative_error)

    def sample_errors(self, pred, truth):
        
        kappa_pred = unpack_sym(pred, dim=self.dim)
        kappa_truth = unpack_sym(truth, dim=self.dim)

        diff = kappa_pred - kappa_truth
        frob_norm_diff = torch.sqrt(torch.sum(diff * diff, dim=(-2, -1)))
        frob_norm_truth = torch.sqrt(torch.sum(kappa_truth * kappa_truth, dim=(-2, -1)))
        relative_error = frob_norm_diff / frob_norm_truth

        return relative_error