import torch


def Piso1(dtype=torch.float64) -> torch.Tensor:
    """Returns the first isotropic projector in Mandel notation."""
    P = torch.zeros((6, 6), dtype=dtype)
    P[:3, :3] = 1. / 3.
    return P

def Piso2(dtype=torch.float64) -> torch.Tensor:
    """Returns the second isotropic projector in Mandel notation."""
    P = torch.eye(6, dtype=dtype)
    P = P - Piso1(dtype=dtype)
    return P

def Ciso(K: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Returns an isotropic stiffness tensor in Mandel notation."""
    P1 = Piso1(dtype=K.dtype).to(K.device)
    I6 = torch.eye(6, dtype=K.dtype).to(K.device)
    
    if K.ndimension() == 1 and G.ndimension() == 1:
        return (3. * K - 2. * G)[:, None, None] * P1[None, :, :] + 2. * G[:, None, None] * I6[None, :, :]
    else:
        return (3. * K - 2. * G) * P1 + 2. * G * I6

def IsoProjectionC(C_mandel):
    if C_mandel.dim() == 2:
        C_mandel = unpack_sym(C_mandel, dim=6)

    K = C_mandel[:, :3, :3].mean(axis=(1, 2))
    G = (torch.diagonal(C_mandel, dim1=1, dim2=2).sum(dim=1) - 3.*K) / 10.
    
    KG_concatenated = torch.stack((K, G), dim=1)
    return  KG_concatenated





# Functions for converting between symmetric matrix representations

def get_sym_indices(dim):
    diag_idx = (torch.arange(dim), torch.arange(dim))    
    row, col = torch.tril_indices(dim, dim, -1)
    dof_idx = (torch.cat([diag_idx[0], row]), torch.cat([diag_idx[1], col]))
    return dof_idx

def pack_sym(symmetric_matrix, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(symmetric_matrix.device) for idx in dof_idx)
    return symmetric_matrix[(..., *dof_idx) if symmetric_matrix.dim() == 3 else dof_idx]

def unpack_sym(packed_values, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(packed_values.device) for idx in dof_idx)
    matrix = torch.zeros((*packed_values.shape[:-1], dim, dim), dtype=packed_values.dtype, device=packed_values.device)
    if packed_values.dim() == 2:
        matrix[:, dof_idx[0], dof_idx[1]] = packed_values
        return matrix + matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(matrix, dim1=1, dim2=2))
    matrix[dof_idx] = packed_values
    return matrix + matrix.T - torch.diag(torch.diag(matrix))