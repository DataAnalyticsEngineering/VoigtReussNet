# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule
from vrnn.data_thermal import Dataset2DThermal, VoigtReussThermNormalization
from vrnn.models import VanillaModule, CNNPlusScalars
from vrnn import utils
import numpy as np
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss
from vrnn.utils import default_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.float32

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
ms_file = data_dir / 'feature_engineering_data.h5'

dtypes = {'images': torch.float32, 'features': torch.float32, 'targets': torch.float32}

# Load data
feature_idx = [0]
R_range_train = [1/100., 1/50., 1/20., 1/10., 1/5., 1/2., 2, 5, 10, 20, 50, 100]
train_data = Dataset2DThermal(ms_file, R_range_train, 'train_set', 
                              input_mode='images', feature_idx=feature_idx, device=device, dtypes=dtypes)
R_range_val = np.concatenate([np.arange(2, 101, dtype=int), 1. / np.arange(2, 101, dtype=int)])
val_data = Dataset2DThermal(ms_file, R_range_val, 'benchmark_set',
                            input_mode='images', feature_idx=feature_idx, device=device, dtypes=dtypes)

# Create dataloaders
batch_size = 5000

# Define normalization
features_max = torch.cat([train_data.features, val_data.features], dim=0).max(dim=0)[0]
features_min = torch.cat([train_data.features, val_data.features], dim=0).min(dim=0)[0]
features_min[0],features_max[0]  = 0, 1 # Dont normalize the first feature (volume fraction)
normalization = VoigtReussThermNormalization(dim=2, features_min=features_min, features_max=features_max)

# For the normalized dataset:
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)

# %%
ann_model = CNNPlusScalars(scalar_input_dim=3, fc_hidden_dims=[256, 128, 64, 32, 16], out_dim=3,
                           conv_channels=[8, 16, 32, 64], kernel_sizes=[13, 9, 7, 5], initial_pool_size=4,
                           img_size=400, n_dim=2)
model_norm = VanillaModule(ann_model).to(device=device, dtype=dtype)

summary(model_norm, input_data=((torch.randn(batch_size, 1, 400, 400, device=device, dtype=dtypes['images']),
                                 torch.randn(batch_size, 3, device=device, dtype=dtypes['features']),),), depth=100)

# %%
loss_fn = VoigtReussNormalizedLoss(dim=2)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=30, min_lr=5e-5)

epochs = 600
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, 
                           val_loader_norm, epochs,
                           verbose=True, scheduler=scheduler)
    
    
fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)
plt.show()

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization)
model = model.to(device=device, dtype=dtype)


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = data_dir / f'Thermal2D_models/IMG_vrnn_therm2D_norm_{current_time}.pth'
torch.save({
    'model_state_dict': model_norm.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, checkpoint_path)
torch.save(model_norm.ann_module, data_dir / f'Thermal2D_models/vrnn_therm2D_img_norm_{current_time}.pt')
torch.save(model, data_dir / f'Thermal2D_models/IMG_vrnn_therm2D_{current_time}.pt')
fig.savefig(data_dir / f'Thermal2D_models/IMG_vrnn_therm2D_training_history_{current_time}.png', dpi=300)


