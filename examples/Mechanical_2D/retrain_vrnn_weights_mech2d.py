# %%
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule, SpectralNormalization
from vrnn.data_mechanical import SpinodoidMechanical2D
from vrnn.models import VanillaModule
from vrnn import utils
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss
from vrnn.utils import default_collate_fn

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

batch_size = 15000

normalization = SpectralNormalization(dim=3, bounds_fn=train_data.calc_bounds)

train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

# %%
# VRNN model
model_norm_file = data_dir / 'Mechanical2D_models/spinodoid_weights_vrnn_20250916_150857.pt'
ann_model = torch.load(model_norm_file, map_location=device, weights_only=False).to(device=device)

ann_model_norm = ann_model.normalized_module
model_norm = VanillaModule(ann_model_norm).to(device=device)

loss_fn = VoigtReussNormalizedLoss(dim=3)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1.56e-03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=50, min_lr=5e-5)

# %%
epochs = 500
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, 
                           val_loader_norm, epochs, verbose=True, scheduler=scheduler)
    
fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)
plt.show()

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Mechanical2D_models/spinodoid_weights_vrnn_{current_time}_retrained.pt')
