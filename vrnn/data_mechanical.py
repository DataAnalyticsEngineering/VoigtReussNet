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
    
class DatasetMechanical2D(Dataset):
    def __init__(
        self,
        file_name: str,
        group: str,
        *,
        contrast_keys: Sequence[str] = ("contrast_inf",),
        input_mode: str = "descriptors",
        feature_idx: Union[slice, Sequence[int], None] = None,
        feature_key: str = "feature_vector",
        image_key: str = "image_data",
        device: Union[str, torch.device] = "cpu",
        dtypes: Mapping[str, torch.dtype] = None,
        periodic_data_augmentation: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        self.file_name = file_name
        self.group = group
        self.contrast_keys = tuple(contrast_keys)
        self.input_mode = input_mode
        self.feature_idx = slice(None) if feature_idx is None else feature_idx
        self.feature_key = feature_key
        self.image_key = image_key
        self.device = torch.device(device)

        if dtypes is None:
            dtypes = {
                "images": torch.float32,
                "features": torch.float32,
                "targets": torch.float32,
            }
        self.dtypes = {k: v for k, v in dtypes.items()}

        self.periodic_data_augmentation = periodic_data_augmentation
        self.max_samples = max_samples

        self.ndim = 2
        self.n_str = 6

        self._load()

    def _load(self):
        with h5py.File(self.file_name, "r") as F:
            desc = torch.tensor(
                F[f"{self.group}/{self.feature_key}"][...],
                dtype=self.dtypes.get("features", torch.float32),
                device=self.device,
            )
            desc = desc[..., self.feature_idx]
            if self.max_samples is not None:
                desc = desc[: self.max_samples]
            n = desc.shape[0]

            if self.input_mode == "images":
                imgs = torch.tensor(
                    F[f"{self.group}/{self.image_key}"][...],
                    dtype=self.dtypes.get("images", torch.float32),
                    device=self.device,
                )
                if self.max_samples is not None:
                    imgs = imgs[: self.max_samples]
                self.images = imgs.unsqueeze(1)
                self.image_height = self.images.shape[-1]

            features, targets = [], []
            for key in self.contrast_keys:
                tgt_path = f"{self.group}/effective_elasticity_tensor/{key}"
                if tgt_path not in F:
                    raise KeyError(
                        f"Dataset key '{tgt_path}' not found in '{self.file_name}'."
                    )
                tgt = torch.tensor(
                    F[tgt_path][...],
                    dtype=self.dtypes.get("targets", torch.float32),
                    device=self.device,
                )
                if self.max_samples is not None:
                    tgt = tgt[: self.max_samples]
                if tgt.shape[0] != n or tgt.shape[1] != self.n_str:
                    raise ValueError(
                        f"Expected target shape ({n}, {self.n_str}), got {tgt.shape}."
                    )

                features.append(desc)
                targets.append(tgt)

            self.features = torch.vstack(features)
            self.targets = torch.vstack(targets)

        self.num_samples = n
        self.num_contrasts = len(self.contrast_keys)
        self.length = self.num_samples * self.num_contrasts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns
        -------
        (features, targets)  if input_mode == 'descriptors'
        ((image, features), targets)  if input_mode == 'images'
        """
        if self.input_mode == "descriptors":
            return self.features[idx], self.targets[idx]

        img_idx = idx % self.num_samples  # same image for every contrast
        if self.periodic_data_augmentation:
            shifts = torch.randint(
                0, self.image_height, (2,), device=self.device
            ).tolist()
            img = torch.roll(
                self.images[img_idx], shifts=shifts, dims=(1, 2)
            )
        else:
            img = self.images[img_idx]

        return (img, self.features[idx]), self.targets[idx]


    def calc_bounds(
        self,
        features: torch.Tensor,
    ):
        if features.shape[-1] < 1:
            raise ValueError("Column 0 must contain the phase-1 volume fraction f₁.")

        f1 = features[..., 0]
        f0 = 1.0 - f1
        # --- build isotropic plane-stress stiffness matrices in Mandel notation -------------- #
        # C = (E / (1 - nu^2)) * [[1, nu, 0],
        #                        [nu, 1, 0],
        #                        [0, 0, (1 - nu)/2 * 2]]
        E0 = torch.tensor(1.0e-8, dtype=features.dtype, device=features.device)
        E1 = torch.tensor(1.0, dtype=features.dtype, device=features.device)
        nu_0 = torch.tensor(0.0, dtype=features.dtype, device=features.device)
        nu_1 = torch.tensor(0.35, dtype=features.dtype, device=features.device)

        fac0 = E0 / (1.0 - nu_0**2)
        fac1 = E1 / (1.0 - nu_1**2)
        C0 = torch.tensor(
            [[1.0, nu_0, 0.0], [nu_0, 1.0, 0.0], [0.0, 0.0, (1.0 - nu_0)]],
            dtype=features.dtype,
            device=features.device,
        ).mul(fac0)
        C1 = torch.tensor(
            [[1.0, nu_1, 0.0], [nu_1, 1.0, 0.0], [0.0, 0.0, (1.0 - nu_1)]],
            dtype=features.dtype,
            device=features.device,
        ).mul(fac1)

        eye = torch.eye(3, dtype=features.dtype, device=features.device)
        C0 = C0.expand(f1.shape + (3, 3))
        C1 = C1.expand(f1.shape + (3, 3))

        upper = f0[..., None, None] * C0 + f1[..., None, None] * C1
        lower = (f0[..., None, None] * torch.linalg.inv(C0) + f1[..., None, None] * torch.linalg.inv(C1)).inverse()

        return lower, upper
