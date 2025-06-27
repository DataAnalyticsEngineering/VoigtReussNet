# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule
from vrnn.data_thermal import DatasetThermal, VoigtReussThermNormalization
from vrnn.models import VanillaModule, MixedActivationMLP
from vrnn import utils
import numpy as np
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.float32

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
ms_file = data_dir / 'feature_engineering_thermal_2D.h5'
print(ms_file)

# Load data
feature_idx = None
R_range_train = [1/100., 1/50., 1/20., 1/10., 1/5., 1/2., 2, 5, 10, 20, 50, 100]
train_data = DatasetThermal(file_name=ms_file, R_range=R_range_train, group='train_set',
                            input_mode='descriptors', feature_idx=feature_idx, feature_key='feature_vector', ndim=2)


R_range_val = np.concatenate([np.arange(2, 101, dtype=int), 1. / np.arange(1, 101, dtype=int)])
val_data = DatasetThermal(file_name=ms_file, R_range=R_range_val, group='val_set',
                          input_mode='descriptors', feature_idx=feature_idx, feature_key='feature_vector', ndim=2)

# Create dataloaders
batch_size = 30000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]


# %%
# Define normalization
features_max = torch.cat([train_data.features, val_data.features], dim=0).max(dim=0)[0]
features_min = torch.cat([train_data.features, val_data.features], dim=0).min(dim=0)[0]
features_min[0],features_max[0]  = 0, 1 # Dont normalize the first feature (volume fraction)
normalization = VoigtReussThermNormalization(dim=2, features_min=features_min, features_max=features_max)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True)


# %%
ann_model = MixedActivationMLP(input_dim=in_dim, hidden_dims=[256, 128, 64, 32, 16], output_dim=out_dim,
                           activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()], 
                           output_activation=nn.Sigmoid(),
                           use_batch_norm=True)

model_norm = VanillaModule(ann_model).to(device=device, dtype=dtype)
print(summary(model_norm, input_size=(batch_size, in_dim), dtypes=[dtype], device=device))

loss_fn = VoigtReussNormalizedLoss(dim=2)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40, min_lr=5e-5)

# %%
epochs = 1000
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, val_loader_norm, epochs,
                         verbose=True, scheduler=scheduler)

fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)


# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization).to(device=device, dtype=dtype)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model_norm.ann_module, data_dir / f'Thermal2D_models/vrnn_therm2D_norm_{current_time}.pt')
torch.save(model, data_dir / f'Thermal2D_models/vrnn_therm2D_{current_time}.pt')

fig.savefig(data_dir / f'Thermal2D_models/vrnn_therm2D_training_history_{current_time}.png', dpi=300)

