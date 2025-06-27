"""
PyTorch data loading
"""
from typing import Optional

import torch
from vrnn.transformations import VoigtReussTransformation
from vrnn.normalization import Normalization
from vrnn.tensortools import Ciso, pack_sym

import numpy as np
import csv
import h5py
from torch.utils.data import Dataset
import os
import random


class VoigtReussMechNormalization(Normalization):
    """
    Voigt-Reuss normalization of a dataset
    """
    def __init__(self,
                 dim: int = 2,
                 features_min: Optional[torch.Tensor] = None,
                 features_max: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.transformation = VoigtReussTransformation(dim=self.dim, **kwargs)
        self.features_min = features_min
        self.features_max = features_max

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each input feature to the range [0, 1].
        
        Parameters:
        x (torch.Tensor): The input tensor to be normalized.
        
        Returns:
        torch.Tensor: The normalized tensor.
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
        """

        """
        f1 = x[..., 0]
        alpha = x[..., -3]
        beta = x[..., -2]
        gamma = x[..., -1]
        kappa0 = Ciso(K=torch.ones_like(beta), G=beta)
        kappa1 = Ciso(K=alpha, G=gamma)

        normalized_y = self.transformation.transform(y, kappa0=kappa0, kappa1=kappa1, f1=f1)

        if verify:
            y_rec = self.transformation.inverse_transform(normalized_y, kappa0=kappa0, kappa1=kappa1, f1=f1)
            
            normalization_errors = torch.linalg.norm(y - y_rec, dim=-1) * 100 / torch.linalg.norm(y, dim=-1)
            recon_violation_entries = torch.where(normalization_errors > 1)[0].tolist()
            for i in recon_violation_entries:
                print(f"Reconstruction violation at index {i}: {normalization_errors[i]}")
                
        return normalized_y

    def reconstruct(self, x, normalized_y, verify=True):
        """

        """
        f1 = x[..., 0]
        alpha = x[..., -3]
        beta = x[..., -2]
        gamma = x[..., -1]
        kappa0 = Ciso(K=torch.ones_like(beta), G=beta)
        kappa1 = Ciso(K=alpha, G=gamma)

        y_rec = self.transformation.inverse_transform(normalized_y, kappa0=kappa0, kappa1=kappa1, f1=f1)
           
        # if verify:
        #     norm_y = self.transformation.transform(y_rec, kappa0=kappa0, kappa1=kappa1, f1=f1)
        #     normalization_errors = torch.linalg.norm(norm_y - normalized_y, dim=-1) * 100 / torch.linalg.norm(normalized_y, dim=-1)
        #     recon_violation_entries = torch.where(normalization_errors > 1)[0].tolist()
        #     for i in recon_violation_entries:
        #         print(f"Reconstruction violation at index {i}: {normalization_errors[i]}")
            
        return y_rec




class Dataset3DMechanical(Dataset):
    def __init__(self, 
                 csv_file_path: str, 
                 h5_file_path: str, 
                 group: str,
                 num_samples: int, 
                 random_seed=42,
                 input_mode='descriptors',
                 feature_idx=None,
                 feature_key='feature_vector',
                 device='cpu',
                 dtypes={'images': torch.float32, 'features': torch.float32, 'targets': torch.float32},
                 periodic_data_augmentation=True):
        """
        A PyTorch Dataset for 3D mechanical microstructure homogenization data.
        
        The final feature vector is formed by:
        [selected_original_features, 1/alpha, 1/beta, 1/gamma, alpha, beta, gamma]

        The homogenized tangent is a 6x6 symmetric matrix. We extract its lower triangle 
        in the order:
        (0,0), (1,1), (2,2), (3,3), (4,4), (5,5),
        (1,0),
        (2,0), (2,1),
        (3,0), (3,1), (3,2),
        (4,0), (4,1), (4,2), (4,3),
        (5,0), (5,1), (5,2), (5,3), (5,4)
        
        Args:
            csv_file_path (str): Path to the CSV file with metadata.
            h5_file_path (str): Path to the HDF5 file with data.
            group (str): 'structures_train', 'structures_val', or 'structures_test'.
            num_samples (int): Number of samples to randomly select.
            feature_vector_name (str): Name of the feature vector dataset in the HDF5.
            random_seed (int): Seed for reproducibility.
            device (str or torch.device): Device for final tensors.
            dtype (torch.dtype): Data type of final tensors.
            feature_idx (None or sequence): Indices of features to keep. If None, keep all.
        """
        # Validate group
        if group not in ['structures_train', 'structures_val', 'structures_test']:
            raise ValueError("group must be one of ['structures_train', 'structures_val', 'structures_test']")

        self.csv_file_path = csv_file_path
        self.h5_file_path = h5_file_path
        self.group = group
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.input_mode = input_mode
        self.feature_idx = feature_idx
        self.feature_key = feature_key
        self.device = device
        self.dtypes = {k: dt for k, dt in dtypes.items()}
        self.periodic_data_augmentation = periodic_data_augmentation
        
        if feature_idx is None:
            feature_idx = slice(None)
        self.feature_idx = feature_idx

        entries = self._load_csv_and_filter()

        if num_samples > len(entries):
            raise ValueError(f"Requested {num_samples} samples, only {len(entries)} available for {group}.")

        random.seed(random_seed)
        self.sampled_entries = random.sample(entries, num_samples)
        self.num_samples = num_samples

        self._load_data()

    def _load_csv_and_filter(self):
        """Load and filter the CSV data for the specified group in a compact manner."""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        target_name = self.group.replace('structures_', '')
        required_fields = ['dataset_index', 'alpha', 'beta', 'gamma', 'hash']
        entries = []
        with open(self.csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['dataset_name'] == target_name:
                    entries.append({
                        field: (int if field == 'dataset_index' else float if field != 'hash' else str)(row[field])
                        for field in required_fields
                    })
        return entries

    def _load_data(self):
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, 'r') as f:
            feature_vector_path = f"/{self.group}/{self.feature_key}"
            if feature_vector_path not in f:
                raise KeyError(f"Feature vector dataset not found at {feature_vector_path}")
            
            feature_vectors = f[feature_vector_path]
            # Truncate feature vector
            feature_vectors = feature_vectors[..., self.feature_idx]

            n_samples = len(self.sampled_entries)
            n_features = feature_vectors.shape[1] + 6  # original features + 6 additional
            features_np = np.empty((n_samples, n_features), dtype=np.float64)
            tangents_np = np.empty((n_samples, 6, 6), dtype=np.float64)
            ms_indices = np.arange(n_samples, dtype=np.int64)

            for idx, entry in enumerate(self.sampled_entries):
                i = entry['dataset_index']
                alpha, beta, gamma = entry['alpha'], entry['beta'], entry['gamma']
                hash_str = entry['hash']
                
                features_np[idx] = np.concatenate([
                    feature_vectors[i,:],
                    [1/alpha, 1/beta, 1/gamma, alpha, beta, gamma]
                ])
                ms_indices[idx] = i              
                tangent_path = f"/{self.group}/dset_{i}/image/{hash_str}/load0/time_step0/homogenized_tangent"
                tangents_np[idx] = f[tangent_path][...]

            # Invert volume fraction of phase 0 to get volume fraction of phase 1
            features_np[:, 0] = 1.0 - features_np[:, 0]
            
            if self.input_mode == 'images':                
                self.img_dtype = self.dtypes.get('images', torch.float32)
                                
                self.h5_file = h5py.File('/dev/shm/FANS_3D_repacked_downscaled.h5', 'r', libver='latest', swmr=True)
                self.images = self.h5_file[f'/{self.group}/microstructure_images']
                self.images = torch.from_numpy(np.expand_dims(self.images, axis=1))
                

        self.ms_indices = ms_indices
        
        # Convert features to torch tensor       
        feat_dtype = self.dtypes.get('features', torch.float32)
        self.features = torch.from_numpy(features_np).to(dtype=feat_dtype, device=self.device)
        
        # Convert targets to torch tensor      
        target_dtype = self.dtypes.get('targets', torch.float32)
        C_all_6x6 = torch.from_numpy(tangents_np).to(dtype=target_dtype, device=self.device)
        self.targets = pack_sym(C_all_6x6, dim=6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.input_mode == 'descriptors':
            return self.features[idx], self.targets[idx]
            
        elif self.input_mode == 'images':
            
            i = self.ms_indices[idx]
            im = self.images[i].to(dtype=self.img_dtype, device=self.device) / 8.0
            
            if self.periodic_data_augmentation:            
                shifts = torch.randint(0, 96, (3,)).tolist() 
                im = torch.roll(im, shifts=shifts, dims=(1, 2, 3))
                
            return (im, self.features[idx]), self.targets[idx]