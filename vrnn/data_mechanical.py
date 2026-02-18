"""
PyTorch data loading
"""
from typing import Sequence, Union, Mapping, Optional
import torch
from vrnn.tensortools import Ciso, pack_sym
import numpy as np
import csv
import h5py
from torch.utils.data import Dataset
import os
import random

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
                 image_key='image_data',
                 device='cpu',
                 dtypes={'images': torch.float64, 'features': torch.float64, 'targets': torch.float64},
                 periodic_data_augmentation=True):
        """
        A PyTorch Dataset for 3D mechanical microstructure homogenization data.
        
        The final feature vector is formed by:
        [selected_original_features, alpha, beta, gamma]

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
        if group not in ['train_set', 'val_set', 'test_set']:
            raise ValueError("group must be one of ['train_set', 'val_set', 'test_set']")

        self.csv_file_path = csv_file_path
        self.h5_file_path = h5_file_path
        self.group = group
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.input_mode = input_mode
        self.feature_idx = feature_idx
        self.feature_key = feature_key
        self.image_key = image_key
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

        target_name = self.group.replace('_set', '')
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
            n_features = feature_vectors.shape[1] + 3  # original features + 3 additional
            features_np = np.empty((n_samples, n_features), dtype=np.float64)
            tangents_np = np.empty((n_samples, 6, 6), dtype=np.float64)
            ms_indices = np.arange(n_samples, dtype=np.int64)

            for idx, entry in enumerate(self.sampled_entries):
                i = entry['dataset_index']
                alpha, beta, gamma = entry['alpha'], entry['beta'], entry['gamma']
                hash_str = entry['hash']
                
                features_np[idx] = np.concatenate([
                    feature_vectors[i,:],
                    [alpha, beta, gamma]
                ])
                ms_indices[idx] = i              
                tangent_path = f"/{self.group}/effective_elasticity_tensor/dset_{i}/microstructure_image_results/{hash_str}/load0/time_step0/homogenized_tangent"
                tangents_np[idx] = f[tangent_path][...]

            # Invert volume fraction of phase 0 to get volume fraction of phase 1
            features_np[:, 0] = 1.0 - features_np[:, 0]
            
            if self.input_mode == 'images':                
                self.img_dtype = self.dtypes.get('images', torch.float32)
                image_path = f"/{self.group}/{self.image_key}"
                
                with h5py.File(self.h5_file_path, 'r') as f:
                    self.images = f[image_path][...]
                    self.images = torch.from_numpy(np.expand_dims(self.images, axis=1))
                    self.images = self.images / 8.0

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
            im = self.images[i].to(dtype=self.img_dtype, device=self.device)
            if self.periodic_data_augmentation:            
                shifts = torch.randint(0, 96, (3,)).tolist() 
                im = torch.roll(im, shifts=shifts, dims=(1, 2, 3))
            return (im, self.features[idx]), self.targets[idx]
    
    def calc_bounds(self, features: torch.Tensor):
        """
        Batched Voigt-Reuss bounds for biphasic 3D isotropic linear elasticity.
        """
        f1   = features[..., 0]
        alpha, beta, gamma = features[..., -3], features[..., -2], features[..., -1]

        C0   = Ciso(K=torch.ones_like(beta), G=beta)
        C1   = Ciso(K=alpha, G=gamma)

        f0   = 1.0 - f1
        lower_bound = (f0[..., None, None] * torch.linalg.inv(C0) + f1[..., None, None] * torch.linalg.inv(C1)).inverse()
        upper_bound =  f0[..., None, None] * C0 + f1[..., None, None] * C1
        
        return lower_bound, upper_bound    

class SpinodoidMechanical2D(Dataset):
    def __init__(
        self,
        file_name: str,
        mode: str,
        frequencies: list[tuple[int, int]],
        split: str,
        target_keys: list[str],
        device: str = "cpu",
        dtypes = {
            "images":   torch.float32,
            "features": torch.float32,
            "targets":  torch.float32,
        },
        max_samples: int | None = None,
        periodic_data_augmentation: bool = True,
        n_vf_augment: int = 0,
    ):
        super().__init__()
        assert mode in {"descriptors", "images", "coeffs"}
        self.mode   = mode
        self.device = device
        self.dtypes = dtypes
        self.periodic_data_augmentation = periodic_data_augmentation

        vfs, ths, tgt_list = [], [], []
        imgs = []     # images mode
        coeffs = []   # coeffs mode (list of [Ni, ki, ki])

        with h5py.File(file_name, "r") as F:
            # ------------ loop over requested frequency groups -------------
            for n in frequencies:
                prefix = f"{split}/mode_grid_{n[0]}x{n[1]}"
                vfs.append(torch.as_tensor(F[f"{prefix}/volume_fraction"][...],
                                           dtype=dtypes["features"]).squeeze())
                ths.append(torch.as_tensor(F[f"{prefix}/threshold"][...],
                                           dtype=dtypes["features"]).squeeze())
                if mode == "images":
                    imgs.append(torch.as_tensor(F[f"{prefix}/image_data"][...],
                                                dtype=dtypes["images"]))
                elif mode == "coeffs" or mode == "descriptors":
                    coeffs.append(torch.as_tensor(F[f"{prefix}/function_weights"][...],
                                                  dtype=dtypes["features"]))  # [Ni, k, k]

            # ------------ load all requested target tensors ----------------
            for key in target_keys:
                tgt_list.append(torch.as_tensor(F[key][...], dtype=dtypes["targets"]))

        # ------------ concatenate along sample dimension -------------------
        self.vf = torch.cat(vfs, dim=0)      # [N]
        self.th = torch.cat(ths, dim=0)      # [N]
        self.y  = pack_sym(torch.cat(tgt_list, dim=0), dim=3)  # [N, 6]

        if self.mode == "images":
            imgs = torch.cat(imgs, dim=0)               # [N, 100, 100]
            self.images = imgs.unsqueeze(1)             # [N,1,100,100]
            self.H = self.images.shape[-1]

        if self.mode == "coeffs" or self.mode == "descriptors":
            # Find global largest kernel (e.g., 11)
            k_max = max(c.shape[1] for c in coeffs)
            # Center-pad each batch from this frequency and then concat
            coeffs = [self._center_pad_batch(c, k_max) for c in coeffs]
            self.A = torch.cat(coeffs, dim=0)           # [N, k_max, k_max]
            self.k_max = k_max
        
        # -------------  data augmentation based on volume fraction range ----------------
        vf_range = [0.2, 0.9]
        orig_vf = self.vf.clone()
        orig_th = self.th.clone()
        orig_y = self.y.clone()
        if self.mode == "images":
            orig_images = self.images.clone()
        if self.mode == "coeffs" or self.mode == "descriptors":
            orig_A = self.A.clone()
        vf_mask = (orig_vf >= vf_range[0]) & (orig_vf <= vf_range[1])
        
        for _ in range(n_vf_augment):
            self.vf = torch.cat([self.vf, orig_vf[vf_mask]], dim=0)
            self.th = torch.cat([self.th, orig_th[vf_mask]], dim=0)
            self.y  = torch.cat([self.y,  orig_y[vf_mask]], dim=0)
            if self.mode == "images":
                self.images = torch.cat([self.images, orig_images[vf_mask]], dim=0)
            if self.mode == "coeffs" or self.mode == "descriptors":
                self.A = torch.cat([self.A, orig_A[vf_mask]], dim=0)

        # -------------  shuffling ---------------------------------
        perm = torch.randperm(self.vf.shape[0], device=self.device)
        self.vf = self.vf.to(self.device)[perm]
        self.th = self.th.to(self.device)[perm]
        self.y  = self.y.to(self.device)[perm]

        if self.mode == "images":
            self.images = self.images.to(self.device)[perm]
        if self.mode == "coeffs" or self.mode == "descriptors":
            self.A  = self.A.to(self.device)[perm]

        # ------------- optional truncation ---------------------------------
        if max_samples is not None:
            self.vf = self.vf[:max_samples]
            self.th = self.th[:max_samples]
            self.y  = self.y [:max_samples]
            if self.mode == "images":
                self.images = self.images[:max_samples]
            if self.mode == "coeffs" or self.mode == "descriptors":
                self.A = self.A[:max_samples]

        self.vf = self.vf.to(device)
        self.th = self.th.to(device)
        self.y  = self.y .to(device)

        self.N = self.vf.shape[0]
        self.features = torch.stack([self.vf, self.th], dim=-1)  # [N, 2]
        self.features_with_coeffs = torch.cat([self.features, self.A.view(self.N, -1)], dim=-1)  # [N, 2 + k_max*k_max]
        self.targets  = self.y       # [N, 6]   

        if self.mode == "images":
            self.images = self.images.to(device)
        if self.mode == "coeffs" or self.mode == "descriptors":
            self.A = self.A.to(device)
        if self.mode == "descriptors":
            self.features = self.features_with_coeffs.to(device)


        # !!!!!! convert targets to Mandel notation !!!!!!!
        self.targets[:,2] *= 2
        self.targets[:,4:] *= np.sqrt(2)
        # !!!!!! convert targets to Mandel notation !!!!!!!

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.mode == "descriptors":
            return self.features[idx], self.targets[idx]

        if self.mode == "images":
            img = self.images[idx]  # [1, H, W]
            if self.periodic_data_augmentation:
                shift = torch.randint(0, img.shape[-1], (2,))
                img   = torch.roll(img, shifts=shift.tolist(), dims=(1, 2))
            return (img, self.features[idx]), self.targets[idx]

        if self.mode == "coeffs" or self.mode == "descriptors":
            return (self.A[idx], self.features[idx]), self.targets[idx]
    
    def _center_pad_batch(self, batch_Amn: torch.Tensor, out_k: int) -> torch.Tensor:
        """
        batch_Amn: [N, m, n] on CPU or GPU, where m and n can be different
        out_k    : int (>= max(m,n))
        returns  : [N, out_k, out_k] with Amn centered
        """
        N, m, n = batch_Amn.shape
        if m == out_k and n == out_k:
            return batch_Amn
        off_m = (out_k - m) // 2
        off_n = (out_k - n) // 2
        out = batch_Amn.new_zeros((N, out_k, out_k))
        out[:, off_m:off_m+m, off_n:off_n+n] = batch_Amn
        return out
        
    def calc_bounds(
        self,
        features: torch.Tensor,
    ):
        if features.shape[-1] < 1:
            raise ValueError("Column 0 must contain the phase-1 volume fraction f₁.")

        f0 = features[..., 0]
        f1 = 1.0 - f0

        E_solid = torch.tensor(1000.0, dtype=features.dtype, device=features.device)
        nu_solid = torch.tensor(0.3, dtype=features.dtype, device=features.device)

        E_void = torch.tensor(1.0, dtype=features.dtype, device=features.device)
        nu_void = torch.tensor(0.49, dtype=features.dtype, device=features.device)
        
        lambda_solid = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
        mu_solid = E_solid / (2 * (1 + nu_solid))
        C0 = torch.tensor([
            [lambda_solid + 2*mu_solid, lambda_solid, 0],
            [lambda_solid, lambda_solid + 2*mu_solid, 0],
            [0, 0, mu_solid]
        ], dtype=features.dtype, device=features.device)

        lambda_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
        mu_void = E_void / (2 * (1 + nu_void))
        C1 = torch.tensor([
            [lambda_void + 2*mu_void, lambda_void, 0],
            [lambda_void, lambda_void + 2*mu_void, 0],
            [0, 0, mu_void]
        ], dtype=features.dtype, device=features.device)

        # !!!!!! convert to mandel notation !!!!!!!
        C0[:,2] *= np.sqrt(2)
        C0[2,:] *= np.sqrt(2)
        C1[:,2] *= np.sqrt(2)
        C1[2,:] *= np.sqrt(2)
        # !!!!!! convert to mandel notation !!!!!!!

        C0 = C0.expand(f0.shape + (3, 3))
        C1 = C1.expand(f1.shape + (3, 3))

        upper = f0[..., None, None] * C0 + f1[..., None, None] * C1
        lower = (f0[..., None, None] * torch.linalg.inv(C0) + f1[..., None, None] * torch.linalg.inv(C1)).inverse()

        return lower, upper
