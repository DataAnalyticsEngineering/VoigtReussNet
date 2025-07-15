# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.data_thermal import DatasetThermal
from vrnn.models import VanillaModule, MixedActivationMLP
from vrnn import utils
from datetime import datetime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
h5_file = data_dir / 'feature_engineering_thermal_3D.h5'

dtypes = {'images': torch.bfloat16, 'features': torch.float32, 'targets': torch.float32}

# Load data
feature_idx = None
R_range_train = [1/100., 1/50., 1/20., 1/10., 1/5., 1/2., 2, 5, 10, 20, 50, 100]
train_data = DatasetThermal(file_name=h5_file, R_range=R_range_train, group='train_set',
                            input_mode='descriptors', feature_idx=feature_idx, feature_key='feature_vector', ndim=3)


R_range_val = np.concatenate([np.arange(2, 101, dtype=int), 1. / np.arange(1, 101, dtype=int)])
val_data = DatasetThermal(file_name=h5_file, R_range=R_range_val, group='val_set',
                          input_mode='descriptors', feature_idx=feature_idx, feature_key='feature_vector', ndim=3,
                          max_samples=2000)

# Create dataloaders
batch_size = 30000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]

# %%
ann_model = MixedActivationMLP(input_dim=in_dim, hidden_dims=[512, 256, 128, 64, 32, 16], output_dim=out_dim,
                               activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()], 
                               output_activation=torch.nn.Identity(),
                               use_batch_norm=True)

model = VanillaModule(ann_model).to(device=device)
print(summary(model, input_size=(batch_size, in_dim), dtypes=[torch.float32], device=device))

# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=40, min_lr=5e-5)

# %%
epochs = 1000
train_losses, val_losses, best_epoch = \
    utils.model_training(model, loss_fn, optimizer, train_loader, val_loader, epochs,
                         verbose=True, scheduler=scheduler)

fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)

# %%

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Thermal3D_models/vann_therm3D_{current_time}.pt')
fig.savefig(data_dir / f'Thermal3D_models/vann_therm3D_training_history_{current_time}.png', dpi=300)
