"""
Various data normalization and parametrization utilities.
"""
import torch
import torch.nn.functional as F
from vrnn.tensortools import get_sym_indices, pack_sym, unpack_sym
from typing import Tuple

class Transformation:
    """
    Abstract transformation class.
    """
    def __init__(self):
        pass

    def transform(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def inverse_transform(self, x):
        raise NotImplementedError("Subclasses must implement this method.")


class IdentityTransformation(Transformation):
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

class OrthogonalTransformation(Transformation):
    """
    Orthogonal transformation class.

    Transforms a batch of orthogonal matrices with shape [..., dim, dim] to
    a batch of degree-of-freedom vectors with shape [..., dim * (dim - 1) / 2] and vice versa.
    """

    def __init__(self, dim: int = 3):
        """
        Constructor of the class.

        :param dim: Physical dimension of the problem, defaults to 3.
        """
        super().__init__()
        self.dim = dim
        self.eye = torch.eye(self.dim)
        self.orth_map = torch.nn.utils.parametrizations._OrthMaps.householder
        self.orth = torch.nn.utils.parametrizations._Orthogonal(
            self.eye, self.orth_map, use_trivialization=False
        )
        stril_indices = torch.tril_indices(self.dim, self.dim, offset=-1)
        self.stril_idx = (stril_indices[0], stril_indices[1])
        self.diag_idx = (list(range(self.dim)), list(range(self.dim)))

    def transform(self, Q):
        """
        Transforms a batch of orthogonal matrices Q with shape [..., dim, dim] to
        a batch of degree-of-freedom vectors with shape [..., dim * (dim - 1) / 2]

        :param Q: Batch of orthogonal matrices with shape [..., dim, dim]
        :return: Batch of degree-of-freedom vectors with shape [..., dim * (dim - 1) / 2]
        """
        # Adjust the shape for the orthogonal parametrization
        self.orth.shape = Q.shape

        # Compute the right inverse to get parameters
        A = self.orth.right_inverse(Q)
        X = A[..., *self.stril_idx]  # Extract strictly lower triangular elements
        
        X = X / 2.0 + 0.5
        return X

    def inverse_transform(self, X):
        """
        Transforms a batch of degree-of-freedom vectors with shape [..., dim * (dim - 1) / 2]
        back to a batch of orthogonal matrices with shape [..., dim, dim]

        :param X: Batch of degree-of-freedom vectors with shape [..., dim * (dim - 1) / 2]
        :return: Batch of orthogonal matrices with shape [..., dim, dim]
        """
        X = 2.0 * (X - 0.5)
        # Initialize A with identity matrices
        A = torch.zeros((*X.shape[:-1], self.dim, self.dim), dtype=X.dtype, device=X.device)
        A[..., *self.diag_idx] = 1.0  # Set diagonal elements to 1
        A[..., *self.stril_idx] = X   # Assign parameters to strictly lower triangular part

        # Compute the orthogonal matrix
        Q = self.orth(A)
        return Q

class VoigtReussTransformation(Transformation):
    """
    Voigt Reuss transformation class.
    """
    def __init__(self, dim: int = 3):
        """
        Constructor of the class.

        :param dim: Physical dimension of the problem, defaults to 3.
        """
        super().__init__()
        self.dim = dim
        self.orth = OrthogonalTransformation(dim=self.dim)
        self.q_trafo = IdentityTransformation()
        self.eig_trafo = IdentityTransformation()

        self.dof_idx = get_sym_indices(self.dim)

    def transform(self, x, kappa_lb, kappa_ub):
        
        kappa = unpack_sym(x, self.dim, self.dof_idx)

        _, L_inv = safe_cholesky(kappa_ub - kappa_lb)
        kappa_norm = L_inv @ (kappa_ub - kappa) @ (L_inv.transpose(-1, -2))
        eig_vals, Q = torch.linalg.eigh(kappa_norm)
        
        # Find violations
        violations = ~((eig_vals >= 0) & (eig_vals <= 1)).all(dim=1)
        failed_entries = torch.where(violations)[0].tolist()
        if failed_entries:
            for i in failed_entries:
                print(f"Eigenvalues of upper bound - lower bound: {torch.linalg.eigvals(kappa_ub[i]-kappa_lb[i]).real.cpu().numpy()}")
                print(f"Eigenvalues of Norm kappa: {eig_vals[i].detach().cpu().numpy()}")
            print(f"Failed entries: {failed_entries}")
        
        eig_vals = torch.clamp(eig_vals, min=0.0, max=1.0)
        
        q_scaled = self.orth.transform(Q)

        normalized_kappa = torch.hstack([self.eig_trafo.transform(eig_vals), self.q_trafo.transform(q_scaled)])

        return normalized_kappa

    def inverse_transform(self, x, kappa_lb, kappa_ub):

        L, _ = safe_cholesky(kappa_ub - kappa_lb)

        eigs_pred = self.eig_trafo.inverse_transform(x[..., :self.dim])

        q_scaled = self.q_trafo.inverse_transform(x[..., self.dim:])
        
        Q = self.orth.inverse_transform(q_scaled)

        eigs_rec = eigs_pred
        kappa_norm = torch.einsum("...ij,...j,...kj->...ik", Q, eigs_rec, Q)

        kappa_pred = kappa_ub - L @ kappa_norm @ L.transpose(-1, -2)
        
        kappa_dof = pack_sym(kappa_pred, self.dim, self.dof_idx)
        
        return kappa_dof

# def safe_cholesky(diff: torch.Tensor, verify: bool = False, rel_tol: float = 1e-9):
    
#     eig_vals, Q = torch.linalg.eigh(diff)
#     abs_tol = torch.sum(eig_vals, dim=-1) * rel_tol
#     mask = eig_vals > abs_tol[:, None]
#     sqrt_eig_vals = torch.zeros_like(eig_vals)
#     sqrt_eig_vals[mask] = torch.sqrt(eig_vals[mask])
#     L = Q * sqrt_eig_vals[:, None, :]
    
#     safe_inv_sqrt_eig_vals = torch.zeros_like(sqrt_eig_vals)
#     safe_inv_sqrt_eig_vals[mask] = 1.0 / sqrt_eig_vals[mask]
#     L_inv = safe_inv_sqrt_eig_vals[:, :, None] * Q.transpose(-2, -1)
         
#     if verify:
#         # Verify reconstruction
#         reconstructed_diff = torch.matmul(L, L.transpose(-1, -2))
#         print(torch.allclose(diff, reconstructed_diff, rtol=1e-1))
        
#         abs_diff = torch.abs(diff - reconstructed_diff)
#         max_diff = torch.max(abs_diff)
#         mean_diff = torch.mean(abs_diff)
#         print(f"Max absolute difference: {max_diff}")
#         print(f"Mean absolute difference: {mean_diff}")
        
#         # Find locations of large differences
#         large_diff_mask = abs_diff > 1e-4
#         if torch.any(large_diff_mask):
#             print("\nLocations with large differences:")
#             large_diff_indices = torch.nonzero(large_diff_mask)
#             for idx in large_diff_indices[:5]:  # Show first 5 differences
#                 print(f"Index {idx}: Original={diff[tuple(idx)]:.6f}, Reconstructed={reconstructed_diff[tuple(idx)]:.6f}")
        
#         print(f"\nAllclose result: {torch.allclose(diff, reconstructed_diff, atol=1e-4)}")
    
#     return L, L_inv


def safe_cholesky(diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
        L      : lower-triangular factor  
        L_inv  : its explicit inverse
    """
    if diff.ndim < 2 or diff.shape[-1] != diff.shape[-2]:
        raise ValueError("diff must be batch of square matrices")

    L = torch.linalg.cholesky(diff) 
    L_inv = torch.linalg.inv(L)    
    return L, L_inv