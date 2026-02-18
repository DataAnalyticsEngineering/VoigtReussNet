
# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule, SpectralNormalization
from vrnn.data_mechanical import SpinodoidMechanical2D
from vrnn.models import VanillaModule, MixedActivationMLP
from vrnn import utils
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dtypes = {'images': torch.float32, 'features': torch.float32, 'targets': torch.float32}

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
h5_file = data_dir / 'feature_engineering_mechanical_2D.h5'

freqs = [(3,3), (5,5), (7,7), (9,9), (11,11)]
# Load data
train_target_keys = [f"/train_set/mode_grid_{i}x{j}/effective_elasticity_tensor/DM8530_TangoBlack" for i, j in freqs]
train_data = SpinodoidMechanical2D(file_name=h5_file, mode = "descriptors", frequencies = freqs, split = "train_set", 
                                   target_keys=train_target_keys, dtypes=dtypes, device=device)

val_target_keys = [f"/val_set/mode_grid_{i}x{j}/effective_elasticity_tensor/DM8530_TangoBlack" for i, j in freqs]
val_data = SpinodoidMechanical2D(file_name=h5_file, mode = "descriptors", frequencies = freqs, split = "val_set", 
                                   target_keys=val_target_keys, dtypes=dtypes, device=device)

# Create dataloaders
batch_size = 15000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]

# Define normalization
features_max = torch.cat([train_data.features, val_data.features], dim=0).max(dim=0)[0]
features_min = torch.cat([train_data.features, val_data.features], dim=0).min(dim=0)[0]
features_min[0],features_max[0]  = 0.0, 1.0 # Dont normalize the first feature (volume fraction)
normalization = SpectralNormalization(dim=3, bounds_fn=train_data.calc_bounds, features_min=features_min, features_max=features_max)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True)

# %%
ann_model = MixedActivationMLP(input_dim=in_dim, hidden_dims=[1024, 512, 256, 128, 128], output_dim=out_dim,
                               activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()], 
                               output_activation=nn.Sigmoid(),
                               use_batch_norm=True)

model_norm = VanillaModule(ann_model).to(device=device)
print(summary(model_norm, input_size=(batch_size, in_dim), dtypes=[dtypes['features']], device=device))


loss_fn = VoigtReussNormalizedLoss(dim=3)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=50, min_lr=5e-5)

# %%
epochs = 1500
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, 
                           val_loader_norm, epochs, verbose=True, scheduler=scheduler)
    
fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)
plt.show()

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Mechanical2D_models/spinodoid_descriptors_vrnn_{current_time}.pt')
fig.savefig(data_dir / f'Mechanical2D_models/spinodoid_descriptors_vrnn_training_history_{current_time}.png', dpi=300)
