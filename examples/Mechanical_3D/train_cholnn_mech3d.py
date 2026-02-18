# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.normalization import NormalizedDataset, NormalizationModule, CholeskyNormalization
from vrnn.data_mechanical import Dataset3DMechanical
from vrnn.models import VanillaModule, MixedActivationMLP
from vrnn import utils
import numpy as np
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtypes = {'features': torch.float64, 'targets': torch.float64, 'images': torch.float64}

# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
h5_file = data_dir / 'feature_engineering_mechanical_3D.h5'
csv_file = data_dir /'metadata_mechanical_3D.csv'

train_data = Dataset3DMechanical(
    csv_file_path= csv_file,
    h5_file_path= h5_file,
    group="train_set",
    num_samples=751089,     # Maximum number of samples: 751089
    random_seed=42,
    input_mode='descriptors',
    feature_idx= None,
    feature_key="feature_vector",
    device=device,
    dtypes=dtypes,
)

val_data = Dataset3DMechanical(
    csv_file_path= csv_file,
    h5_file_path= h5_file,
    group="val_set",
    num_samples=263188,   # Maximum number of samples: 263188
    random_seed=42,
    input_mode='descriptors',
    feature_idx= None,
    feature_key="feature_vector",
    device=device,
    dtypes=dtypes,
)

# Create dataloaders
batch_size = 75000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]


# %%
# Define normalization
features_max = torch.cat([train_data.features, val_data.features], dim=0).max(dim=0)[0]
features_min = torch.cat([train_data.features, val_data.features], dim=0).min(dim=0)[0]
features_min[0],features_max[0]  = 0.0, 1.0 # Dont normalize the first feature (volume fraction)
normalization = CholeskyNormalization(dim=6, features_min=features_min, features_max=features_max)

# Normalize data
train_data_norm = NormalizedDataset(train_data, normalization)
val_data_norm = NormalizedDataset(val_data, normalization)
train_loader_norm = DataLoader(train_data_norm, batch_size=batch_size, shuffle=False)
val_loader_norm = DataLoader(val_data_norm, batch_size=batch_size, shuffle=False)
# %%
ann_model = MixedActivationMLP(input_dim=in_dim, hidden_dims=[1024, 512, 256, 128, 128], output_dim=out_dim,
                               activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()], 
                               output_activation=torch.nn.Identity(),
                               use_batch_norm=True)

model_norm = VanillaModule(ann_model).to(device=device, dtype=dtypes['features'])
print(summary(model_norm, input_size=(batch_size, in_dim), dtypes=[dtypes['features']], device=device))

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=5e-5)

# %%
epochs = 2000
train_losses, val_losses, best_epoch = \
    utils.model_training(model_norm, loss_fn, optimizer, train_loader_norm, val_loader_norm, epochs,
                         verbose=True, scheduler=scheduler)

fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)

# %%
model = NormalizationModule(normalized_module=model_norm, normalization=normalization).to(device=device)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Mechanical3D_models/cholnn_mech3D_{current_time}.pt')
fig.savefig(data_dir / f'Mechanical3D_models/cholnn_mech3D_training_history_{current_time}.png', dpi=300)
