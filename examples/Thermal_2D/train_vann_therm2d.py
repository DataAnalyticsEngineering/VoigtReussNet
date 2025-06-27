# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from vrnn.data_thermal import Dataset2DThermal
from vrnn.models import VanillaModule, MixedActivationLayer
from vrnn.tensortools import unpack_sym
from vrnn import utils
import numpy as np
from datetime import datetime

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.float32


# %%
# Load hdf5 files
data_dir = utils.get_data_dir()
ms_file = data_dir / 'feature_engineering_data.h5'
print(ms_file)

# Load data
feature_idx = None
R_range_train = [1/100., 1/50., 1/20., 1/10., 1/5., 1/2., 2, 5, 10, 20, 50, 100]
train_data = Dataset2DThermal(ms_file, R_range_train, 'train_set', feature_idx=feature_idx)
R_range_val = np.concatenate([np.arange(2, 101, dtype=int), 1. / np.arange(2, 101, dtype=int)])
val_data = Dataset2DThermal(ms_file, R_range_val, 'benchmark_set', feature_idx=feature_idx)

# Create dataloaders
batch_size = 30000
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

in_dim, out_dim = train_data.features.shape[-1], train_data.targets.shape[-1]

# %%
# Model parameters
hidden_dims = [256, 128, 64, 32, 16]
drop_probs = [0.0, 0.0, 0.0, 0.0, 0.0]
activation_functions = [ 
    [torch.nn.SELU(), torch.nn.Tanh(), torch.nn.Sigmoid(), torch.nn.Identity()],
    [torch.nn.SELU(), torch.nn.Tanh(), torch.nn.Sigmoid(), torch.nn.Identity()],
    [torch.nn.SELU(), torch.nn.Tanh(), torch.nn.Sigmoid(), torch.nn.Identity()],
    [torch.nn.SELU(), torch.nn.Tanh(), torch.nn.Sigmoid(), torch.nn.Identity()],
    [torch.nn.SELU(), torch.nn.Tanh(), torch.nn.Sigmoid(), torch.nn.Identity()],
]

layers = []
prev_dim = in_dim
for i, hidden_dim in enumerate(hidden_dims):
    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
    layers.append(torch.nn.BatchNorm1d(hidden_dim))
    layers.append(MixedActivationLayer(activation_functions[i]))
    layers.append(torch.nn.Dropout(drop_probs[i]))
    prev_dim = hidden_dim

layers.append(torch.nn.Linear(prev_dim, out_dim))

ann_model = torch.nn.Sequential(*layers).to(device=device, dtype=dtype)
model = VanillaModule(ann_model).to(device=device, dtype=dtype)
print(summary(model, input_size=(batch_size, in_dim), dtypes=[dtype], device=device))

l1_loss = nn.L1Loss(reduction="mean")
mse_loss = nn.MSELoss(reduction="mean")
def combined_loss(pred, target):
    return l1_loss(pred, target) + mse_loss(pred, target)
loss_fn = combined_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=5e-4)

# %%
epochs = 750
train_losses, val_losses, best_epoch = \
    utils.model_training(model, loss_fn, optimizer, train_loader, val_loader, epochs,
                         verbose=True, scheduler=scheduler)

fig, ax = plt.subplots()
utils.plot_training_history(ax, train_losses, val_losses, best_epoch)

# %%
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model, data_dir / f'Thermal2D_models/vann_therm2D_{current_time}.pt')

fig.savefig(data_dir / f'Thermal2D_models/vann_therm2D_training_history_{current_time}.png', dpi=300)

# %%
# Fetch data and compute errors
train_x, train_y = utils.get_data(train_loader, device=device, dtype=dtype)
val_x, val_y = utils.get_data(val_loader, device=device, dtype=dtype)

with torch.inference_mode():
    train_pred = model(train_x).squeeze(0)
    val_pred = model(val_x).squeeze(0)
    train_loss = loss_fn(train_pred, train_y)
    val_loss = loss_fn(val_pred.squeeze(0), val_y.squeeze(0))
    print(f'train loss {train_loss:e}, validation loss {val_loss:e}')


rel_err_train = torch.norm(unpack_sym(train_y, dim=2) - unpack_sym(train_pred, dim=2), 'fro', dim=(1,2)) * 100 / torch.norm(unpack_sym(train_y, dim=2), 'fro', dim=(1,2))
rel_err_val = torch.norm(unpack_sym(val_y,dim=2) - unpack_sym(val_pred,dim=2), 'fro', dim=(1,2)) * 100 / torch.norm(unpack_sym(val_y,dim=2), 'fro', dim=(1,2))

print(f'median rel. frobenius error (%) (training) {rel_err_train.median():.4f}, '
      f'median rel. frobenius error (%) (validation) {rel_err_val.median():.4f}')
print(f'mean rel. frobenius error (%) (training) {rel_err_train.mean():.4f}, '
      f'mean rel. frobenius error (%) (validation) {rel_err_val.mean():.4f}')
print(f'max rel. frobenius error (%) (training) {rel_err_train.max():.4f}, '
      f'max rel. frobenius error (%) (validation) {rel_err_val.max():.4f}')
print(f'min rel. frobenius error (%) (training) {rel_err_train.min():.4f}, '
      f'min rel. frobenius error (%) (validation) {rel_err_val.min():.4f}')
print(f'std rel. frobenius error (%) (training) {rel_err_train.std():.4f}, '
      f'std rel. frobenius error (%) (validation) {rel_err_val.std():.4f}')


