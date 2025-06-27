# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule
from vrnn.data_thermal import Dataset3DThermal, VoigtReussThermNormalization
from vrnn.models import VanillaModule, MixedActivationMLP
from vrnn import utils
from datetime import datetime
from vrnn.losses import VoigtReussNormalizedLoss
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = utils.get_data_dir()

# %%
# Load hdf5 files
h5_file = data_dir / 'FANS_3D.h5'

dtypes = {'images': torch.bfloat16, 'features': torch.float64, 'targets': torch.float64}

# Load data
feature_idx = None
R_range_train = [1/100., 1/5., 5, 100]
train_data = Dataset3DThermal(file_name=h5_file, 
                              R_range=R_range_train, 
                              group='structures_train', 
                              input_mode='descriptors',
                              feature_idx=feature_idx, 
                              device=device,
                              dtypes=dtypes,
                              max_samples=12600)
# R_range_val = np.concatenate([np.arange(2, 101, dtype=int), 1. / np.arange(2, 101, dtype=int)])
R_range_val = [1/100., 1/5., 5, 100]
val_data = Dataset3DThermal(file_name=h5_file,
                            R_range=R_range_val,
                            group='structures_val',
                            input_mode='descriptors',
                            feature_idx=feature_idx,
                            device=device,
                            dtypes=dtypes,
                            max_samples=20)

# Create dataloaders
batch_size = 12600
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# %%
# Define normalization
features_max = torch.cat([train_data.features, val_data.features], dim=0).max(dim=0)[0]
features_min = torch.cat([train_data.features, val_data.features], dim=0).min(dim=0)[0]
features_min[0],features_max[0]  = 0.0, 1.0 # Dont normalize the first feature (volume fraction)
normalization = VoigtReussThermNormalization(dim=3, features_min=features_min, features_max=features_max)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)

train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=True)

# %%
ann_model = MixedActivationMLP(input_dim=238, hidden_dims=[512, 256, 128, 64, 32, 16], output_dim=6,
                               activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()], 
                               output_activation=nn.Sigmoid(),
                               use_batch_norm=True)
model_norm = VanillaModule(ann_model).to(device=device)

summary(model_norm, input_size=(1, 238))

# %%
loss_fn = VoigtReussNormalizedLoss(dim=3)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=30, min_lr=5e-5)

epochs = 1000
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, val_loader_norm, epochs,
                         verbose=True, scheduler=scheduler)

fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization).to(device=device)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = data_dir / f'Thermal3D_models/sweep_over_num_samples/vrnn6per_therm3D_norm_{current_time}.pth'
torch.save({
    'model_state_dict': model_norm.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, checkpoint_path)


torch.save(model, data_dir / f'Thermal3D_models/sweep_over_num_samples/vrnn6per_therm3D_{current_time}.pt')

fig.savefig(data_dir / f'Thermal3D_models/sweep_over_num_samples/vrnn6per_therm3D_training_history_{current_time}.png', dpi=300)


