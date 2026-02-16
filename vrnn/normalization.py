"""
Normalization
"""
import torch
from torch.utils.data import Dataset
from vrnn.base import BaseModule
from typing import Optional
from vrnn.transformations import VoigtReussTransformation
from vrnn.tensortools import unpack_sym, pack_sym

class Normalization:
    """
    Abstract normalization class.
    Subclasses must implement:
      - normalize_x: to normalize the input features
      - normalize_y: to normalize the target y (possibly using x)
      - reconstruct: to reconstruct the predictions back to the original scale
    """
    def __init__(self):
        pass

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def normalize_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def reconstruct(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class NormalizedDataset(Dataset):
    """
    Wraps a cached dataset and precomputes normalized x and y.
    
    For descriptors mode, x is a tensor (features).
    For images mode, x is a tuple (image, features), but only the features are normalized.
    The normalized values are computed in one vectorized call.
    """
    def __init__(self, dataset: Dataset, normalization):
        """
        normalization: an object (e.g. your VoigtReussThermNormalization instance)
                       that provides vectorized methods:
                         - normalize_x(tensor)  and 
                         - normalize_y(x, y)
        """
        self.normalization = normalization
        self.length = len(dataset)
        self.dataset = dataset
        x0, _ = self.dataset[0]
        self.mode = 'images' if isinstance(x0, (tuple, list)) else 'descriptors'
        self._precompute_normalized_values(dataset)
        
    def _precompute_normalized_values(self, dataset):
        
        # Check if dataset has features and targets attributes
        if not hasattr(dataset, 'features') or not hasattr(dataset, 'targets'):
            raise AttributeError("Dataset must have 'features' and 'targets' attributes")

        self.features = self.normalization.normalize_x(dataset.features)
        self.targets = self.normalization.normalize_y(dataset.features, dataset.targets)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'descriptors':
            return self.features[idx], self.targets[idx]
        elif self.mode == 'images':
            (im, _), _ = self.dataset[idx]
            return (im, self.features[idx]), self.targets[idx]


class NormalizationModule(BaseModule):
    """
    Wraps a module so that the input features are first normalized and then passed
    to the module. The module's output is then converted back to the original scale
    using the normalization's reconstruct method.
    
    In descriptors mode, x is a tensor.
    In images mode, x is a tuple: (image, features) and only the features are normalized.
    """
    def __init__(self, normalized_module: torch.nn.Module, normalization: Normalization):
        super().__init__()
        self.normalized_module = normalized_module
        self.normalization = normalization

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            image, features = x
            norm_features = self.normalization.normalize_x(features)
            normalized_input = (image, norm_features)
            normalized_pred = self.normalized_module(normalized_input)
            pred = self.normalization.reconstruct(features, normalized_pred)
        else:
            normalized_input = self.normalization.normalize_x(x)
            normalized_pred = self.normalized_module(normalized_input)
            pred = self.normalization.reconstruct(x, normalized_pred)

        return pred

class SpectralNormalization(Normalization):
    def __init__(self,
                 bounds_fn,
                 dim: int = 2,
                 features_min: Optional[torch.Tensor] = None,
                 features_max: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.transformation = VoigtReussTransformation(dim=self.dim, **kwargs)
        self.features_min = features_min
        self.features_max = features_max
        
        if bounds_fn is None:
            raise ValueError("SpectralNormalization needs a function to compute spectral bounds.")
        self.bounds_fn = bounds_fn

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each input feature to the range [0, 1].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x.dtype, device=x.device)
            self.features_max = self.features_max.to(dtype=x.dtype, device=x.device)
            features_norm = (x - self.features_min) / (self.features_max - self.features_min)
        else:
            features_norm = x

        return features_norm

    def normalize_y(self, x, y, verify=True):
        lower_bound, upper_bound = self.bounds_fn(x)
        normalized_y = self.transformation.transform(y, lower_bound, upper_bound)
        
        if verify:
            y_rec = self.transformation.inverse_transform(normalized_y, lower_bound, upper_bound)
            
            normalization_errors = torch.linalg.norm(y - y_rec, dim=-1) * 100 / torch.linalg.norm(y, dim=-1)
            recon_violation_entries = torch.where(normalization_errors > 1)[0].tolist()
            for i in recon_violation_entries:
                print(f"Reconstruction violation at index {i}: {normalization_errors[i]}")
                
        return normalized_y

    def reconstruct(self, x, normalized_y):
        lower_bound, upper_bound = self.bounds_fn(x)
        y_rec      = self.transformation.inverse_transform(normalized_y, lower_bound, upper_bound)
        
        return y_rec
    
    def reconstruct_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original input features from normalized [0, 1] range.
        """
        if not isinstance(x_norm, torch.Tensor):
            raise TypeError("Input x_norm must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x_norm.dtype, device=x_norm.device)
            self.features_max = self.features_max.to(dtype=x_norm.dtype, device=x_norm.device)
            x_reconstructed = x_norm * (self.features_max - self.features_min) + self.features_min
        else:
            x_reconstructed = x_norm

        return x_reconstructed


class CholeskyNormalization(Normalization):
    def __init__(self,
                 dim: int = 2,
                 features_min: Optional[torch.Tensor] = None,
                 features_max: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.features_min = features_min
        self.features_max = features_max
        
    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each input feature to the range [0, 1].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x.dtype, device=x.device)
            self.features_max = self.features_max.to(dtype=x.dtype, device=x.device)
            features_norm = (x - self.features_min) / (self.features_max - self.features_min)
        else:
            features_norm = x

        return features_norm

    def normalize_y(self, x, y, verify=True):
        y_mat = unpack_sym(y, dim=self.dim)
        
        L = torch.linalg.cholesky(y_mat)
        normalized_y = pack_sym(L, dim=self.dim)
        
        
        if verify:
            y_rec = pack_sym(L @ L.transpose(-1, -2), dim=self.dim)
            
            normalization_errors = torch.linalg.norm(y - y_rec, dim=-1) * 100 / torch.linalg.norm(y, dim=-1)
            recon_violation_entries = torch.where(normalization_errors > 1)[0].tolist()
            for i in recon_violation_entries:
                print(f"Reconstruction violation at index {i}: {normalization_errors[i]}")
                
        return normalized_y

    def reconstruct(self, x, normalized_y):
        L = unpack_sym(normalized_y, dim=self.dim)
        
        # Ensure L is lower triangular (zero out upper triangular elements)
        mask = torch.tril(torch.ones(self.dim, self.dim, device=normalized_y.device, dtype=normalized_y.dtype))
        for _ in range(len(L.shape) - 2):
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(L)
        L = L * mask
        y_rec = L @ L.transpose(-1, -2)
        y_rec = pack_sym(y_rec, dim=self.dim)
        
        return y_rec
    
    def reconstruct_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original input features from normalized [0, 1] range.
        """
        if not isinstance(x_norm, torch.Tensor):
            raise TypeError("Input x_norm must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x_norm.dtype, device=x_norm.device)
            self.features_max = self.features_max.to(dtype=x_norm.dtype, device=x_norm.device)
            x_reconstructed = x_norm * (self.features_max - self.features_min) + self.features_min
        else:
            x_reconstructed = x_norm

        return x_reconstructed
    
class VanillaNormalization(Normalization):
    def __init__(self,
                 dim: int = 2,
                 features_min: Optional[torch.Tensor] = None,
                 features_max: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.features_min = features_min
        self.features_max = features_max
        
    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each input feature to the range [0, 1].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x.dtype, device=x.device)
            self.features_max = self.features_max.to(dtype=x.dtype, device=x.device)
            features_norm = (x - self.features_min) / (self.features_max - self.features_min)
        else:
            features_norm = x

        return features_norm

    def normalize_y(self, x, y, verify=True):
        normalized_y = y
                
        return normalized_y

    def reconstruct(self, x, normalized_y):
        y_rec = normalized_y
        
        return y_rec
    
    def reconstruct_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original input features from normalized [0, 1] range.
        """
        if not isinstance(x_norm, torch.Tensor):
            raise TypeError("Input x_norm must be a torch.Tensor")
        if (self.features_min is not None) and (self.features_max is not None):
            self.features_min = self.features_min.to(dtype=x_norm.dtype, device=x_norm.device)
            self.features_max = self.features_max.to(dtype=x_norm.dtype, device=x_norm.device)
            x_reconstructed = x_norm * (self.features_max - self.features_min) + self.features_min
        else:
            x_reconstructed = x_norm

        return x_reconstructed