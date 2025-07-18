"""
PyTorch data loading
"""
import h5py
import torch
from torch.utils.data import Dataset

class DatasetThermal(Dataset):
    def __init__(self, file_name, R_range, group, 
                 input_mode='descriptors', 
                 feature_idx=None,
                 feature_key='feature_vector', 
                 image_key='image_data',
                 device='cpu',
                 dtypes={'images': torch.float32, 'features': torch.float32, 'targets': torch.float32},
                 periodic_data_augmentation=True,
                 max_samples=None,
                 ndim=2):
        """
        A PyTorch Dataset for thermal microstructure data.
        """
        self.file_name = file_name
        self.R_range = R_range
        self.group = group
        self.input_mode = input_mode
        self.feature_idx = slice(None) if (feature_idx is None) else feature_idx
        self.image_key = image_key
        self.feature_key = feature_key
        self.num_R = len(R_range)
        self.device = device
        self.dtypes = {k: dt for k, dt in dtypes.items()}
        self.periodic_data_augmentation = periodic_data_augmentation
        self.max_samples = max_samples
        
        self.ndim = ndim
        self.n_str = int(ndim*(ndim + 1)/2)
        
        self._load()
        
    def _load(self):
        with h5py.File(self.file_name, "r") as F:
            descriptors = torch.tensor(F[f"{self.group}/{self.feature_key}"][...],
                                       dtype=self.dtypes.get('features', torch.float32),
                                       device=self.device)
            descriptors = descriptors[..., self.feature_idx]
            if self.max_samples is not None:    # Optionally limit the dataset size (max_samples)
                descriptors = descriptors[: self.max_samples]
            num_samples = descriptors.shape[0]
            features_list = []
            kappa_list = []
            
            for R in self.R_range:
                feat_dtype = self.dtypes.get('features', torch.float32)
                R_column = torch.full((num_samples, 1), R, dtype=feat_dtype, device=self.device)
                onebyR_column = torch.full((num_samples, 1), 1.0 / R, dtype=feat_dtype, device=self.device)
                feat_R = torch.hstack((descriptors, onebyR_column, R_column))
                features_list.append(feat_R)
                
                if R == 1:
                    kappa_R = torch.ones((num_samples, self.n_str), dtype=self.dtypes.get('targets', torch.float32), device=self.device)
                    kappa_R[:, self.ndim:] = 0.0
                else:
                    R_key = int(round(1 / R)) if R < 1 else int(round(R))
                    key_suffix = "_invR" if R < 1 else "_R"
                    key = f"{self.group}/effective_heat_conductivity/contrast{key_suffix}_{R_key}"
                    if key not in F:
                        raise KeyError(f"Dataset key '{key}' not found in file '{self.file_name}'.")
                    kappa_R = torch.tensor(F[key][...],
                                           dtype=self.dtypes.get('targets', torch.float32),
                                           device=self.device)
                    if self.max_samples is not None:   # Optionally limit the dataset size (max_samples)
                        kappa_R = kappa_R[: self.max_samples]
                    if kappa_R.shape[0] != num_samples or kappa_R.shape[1] != self.n_str:
                        raise ValueError(f"Expected kappa shape ({num_samples}, {self.n_str}), got {kappa_R.shape}.")
                kappa_list.append(kappa_R)
                
            self.features = torch.vstack(features_list)
            if self.ndim == 3:
                self.features[:,0] = 1.0 - self.features[:,0]
            self.targets = torch.vstack(kappa_list)
            target_dtype = self.dtypes.get('targets', torch.float32)
            self.targets[:,self.ndim:] /= torch.sqrt(torch.tensor(2.0, dtype=target_dtype, device=self.device))
            self.num_samples = num_samples
            
            if self.input_mode == 'images':
                images = torch.tensor(F[f"{self.group}/{self.image_key}"][...],
                                      dtype=self.dtypes.get('images', torch.float32),
                                      device=self.device)
                self.images = images.unsqueeze(1)
                self.image_height = self.images.shape[-1]
                
        self.length = self.num_samples * self.num_R

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.input_mode == 'descriptors':
            return self.features[idx], self.targets[idx]
        elif self.input_mode == 'images':
            image_idx = idx % self.num_samples

            if self.periodic_data_augmentation:            
                shifts = torch.randint(0, self.image_height, (2,)).tolist()        
                return ((torch.roll(self.images[image_idx], shifts=shifts, dims=(1, 2)), 
                        self.features[idx]), self.targets[idx])
            else:
                return (self.images[image_idx], self.features[idx]), self.targets[idx]
    
    def calc_bounds(self, features: torch.Tensor):
        """
        Batched Voigt-Reuss bounds for biphasic 2D/3D isotropic thermal conductivity.
        """
        f1 = features[..., 0]
        R  = features[..., -1, None, None]       
        dim = self.ndim

        eye = torch.eye(dim, dtype=features.dtype, device=features.device)
        kappa0  = eye
        kappa1  = (1.0 / R) * eye

        f0  = 1.0 - f1
        lower_bound = (f0[..., None, None] * torch.linalg.inv(kappa0) + f1[..., None, None] * torch.linalg.inv(kappa1)).inverse()
        upper_bound = f0[..., None, None] * kappa0 + f1[..., None, None] * kappa1
        
        return lower_bound, upper_bound
