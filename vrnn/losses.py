import torch
from vrnn.transformations import OrthogonalTransformation
import torch.nn as nn

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