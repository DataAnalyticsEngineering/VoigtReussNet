# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule, SpectralNormalization
from vrnn.data_mechanical import DatasetMechanical2D
from vrnn.models import VanillaModule, CNNPlusScalars
from vrnn import utils
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss
from vrnn.utils import default_collate_fn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dtypes = {'images': torch.float32, 'features': torch.float32, 'targets': torch.float32}

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
h5_file = data_dir / 'orthotropic_plane_stress_mechanical_2D.h5'

# Load data
train_data = DatasetMechanical2D(file_name=h5_file, group='train_set', contrast_keys = ("contrast_inf",),
                                 input_mode='images', feature_idx=None, feature_key='feature_vector',
                                 image_key="image_data", periodic_data_augmentation=True, device=device)

val_data = DatasetMechanical2D(file_name=h5_file, group='val_set', contrast_keys = ("contrast_inf",),
                               input_mode='images', feature_idx=None, feature_key='feature_vector',
                               image_key="image_data", periodic_data_augmentation=True, device=device)

# Create dataloaders
batch_size = 30000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]

# Define normalization
normalization = SpectralNormalization(dim=3, bounds_fn=train_data.calc_bounds)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

# %%
ann_model = CNNPlusScalars(scalar_input_dim=1, fc_hidden_dims=[128, 64, 32, 16], out_dim=6, conv_channels=[4, 8, 16],
                           kernel_sizes=[3, 3, 3], initial_pool_size=1, img_size=50, n_dim=2)

model_norm = VanillaModule(ann_model).to(device=device)
print(summary(model_norm, input_data=((torch.randn(batch_size, 1, 50, 50, device=device, dtype=dtypes['images']),
                                 torch.randn(batch_size, 1, device=device, dtype=dtypes['features']),),), depth=100))

loss_fn = VoigtReussNormalizedLoss(dim=3)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=40, min_lr=5e-5)

# %%
epochs = 1000
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, 
                           val_loader_norm, epochs, verbose=True, scheduler=scheduler)
    
fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)
plt.show()

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Mechanical2D_models/orthotropic_vrnn_{current_time}.pt')
fig.savefig(data_dir / f'Mechanical2D_models/orthotropic_vrnn_training_history_{current_time}.png', dpi=300)
