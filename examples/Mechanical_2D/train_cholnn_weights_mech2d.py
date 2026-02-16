# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule, CholeskyNormalization
from vrnn.data_mechanical import SpinodoidMechanical2D
from vrnn.models import VanillaModule, CNNPlusScalars, RenderWrap2D
from vrnn import utils
from datetime import datetime
from vrnn.utils import default_collate_fn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dtypes = {'images': torch.float32, 'features': torch.float32, 'targets': torch.float32}

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
h5_file = data_dir / 'spinodoids_plane_strain_mechanical_2D.h5'

# Load data
train_target_keys = [f"/train_set/frequency_{i}/effective_elasticity_tensor/DM8530_TangoBlack" for i in range(1, 6)]
train_data = SpinodoidMechanical2D(file_name=h5_file, mode = "coeffs", frequencies = [1,2,3,4,5], split = "train_set", 
                                   target_keys=train_target_keys, dtypes=dtypes, device=device)

val_target_keys = [f"/val_set/frequency_{i}/effective_elasticity_tensor/DM8530_TangoBlack" for i in range(1, 6)]
val_data = SpinodoidMechanical2D(file_name=h5_file, mode = "coeffs", frequencies = [1,2,3,4,5], split = "val_set", 
                                   target_keys=val_target_keys, dtypes=dtypes, device=device)
# Create dataloaders
batch_size = 15000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]

# Define normalization
normalization = CholeskyNormalization(dim=3)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

# %%
base_cnn = CNNPlusScalars(scalar_input_dim=in_dim, fc_hidden_dims=[256, 128, 64, 64, 64], out_dim=out_dim, conv_channels=[18, 24, 30, 36],
                           kernel_sizes=[3, 5, 7, 9], initial_pool_size=4, img_size=400, n_dim=2,
                           output_activation=nn.Identity())
ann_model = RenderWrap2D(base_cnn=base_cnn, img_size=(400, 400), k_size=11, temp=0.0001, periodic_shift=False)

model_norm = VanillaModule(ann_model).to(device=device)
print(summary(model_norm, input_data=((torch.randn(batch_size, 11, 11, device=device, dtype=dtypes['images']),
                                 torch.randn(batch_size, in_dim, device=device, dtype=dtypes['features']),),), depth=100))

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=50, min_lr=5e-5)

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
torch.save(model, data_dir / f'Mechanical2D_models/spinodoids/spinodoid_weights_cholnn_{current_time}.pt')
fig.savefig(data_dir / f'Mechanical2D_models/spinodoids/spinodoid_weights_cholnn_training_history_{current_time}.png', dpi=300)
