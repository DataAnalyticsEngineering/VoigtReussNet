"""
Normalization
"""
import torch
from torch.utils.data import Dataset
from vrnn.base import BaseModule 


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

        self.normalized_x = self.normalization.normalize_x(dataset.features)
        self.normalized_y = self.normalization.normalize_y(dataset.features, dataset.targets)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'descriptors':
            return self.normalized_x[idx], self.normalized_y[idx]
        elif self.mode == 'images':
            (im, _), _ = self.dataset[idx]
            return (im, self.normalized_x[idx]), self.normalized_y[idx]


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
