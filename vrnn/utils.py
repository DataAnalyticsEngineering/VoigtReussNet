import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as f


def get_file_dir():
    """[summary]

    :return: [description]
    :rtype: [type]
    """
    return Path(__file__).resolve().parent


def get_data_dir():
    """[summary]

    :return: [description]
    :rtype: [type]
    """
    return get_file_dir().parent / 'data'


def get_data(data_loader, device='cpu', dtype=None):
    """[summary]

    :param data_loader: [description]
    :type data_loader: [type]
    :param device: [description], defaults to 'cpu'
    :type device: str, optional
    :return: [description]
    :rtype: [type]
    """
    if dtype is None:
        dtype = torch.float32

    x_list, y_list = [], []
    for x_batch, y_batch in list(data_loader):
        x_list.append(x_batch)
        y_list.append(y_batch)
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    return x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)

def model_training(model, loss_fn, optimizer, train_loader, val_loader, epochs, verbose=False, scheduler=None):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param loss_fn: [description]
    :type loss_fn: [type]
    :param optimizer: [description]
    :type optimizer: [type]
    :param train_loader: [description]
    :type train_loader: [type]
    :param val_loader: [description]
    :type val_loader: [type]
    :param epochs: [description]
    :type epochs: [type]
    :param verbose: [description], defaults to False
    :type verbose: bool, optional
    :raises Exception: [description]
    :return: [description]
    :rtype: [type]
    """
    early_stop_patience = 1
    early_stop_counter = early_stop_patience
    epoch_list = []
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_loss = float('inf')
    best_parameters = model.state_dict()
    for t in range(epochs):
        epoch_list.append(t + 1)
        # training step:
        train_loss = model.training_step(train_loader, loss_fn, optimizer)
        # train_loss = model.loss_calculation(train_loader, loss_fn)
        train_losses.append(train_loss)
        if np.isnan(train_loss):
            raise Exception('training loss is not a number')
        # validation step:
        val_loss = model.loss_calculation(val_loader, loss_fn)
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        # early stopping:
        if t > int(0.1 * epochs) and val_loss < best_loss:
            if early_stop_counter < early_stop_patience:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
                best_epoch, best_loss = t, val_loss
                best_parameters = model.state_dict()
        # status update:
        if verbose and ((t + 1) % 1 == 0):
            print(f"Epoch {t + 1}: training loss {train_loss:>8f}, validation loss {val_loss:>8f}, learning rate {scheduler.optimizer.param_groups[0]['lr']:.2e}")
    model.load_state_dict(best_parameters)
    return train_losses, val_losses, best_epoch


def plot_training_history(ax, train_losses, val_losses, best_epoch):
    """[summary]

    :param train_losses: [description]
    :type train_losses: [type]
    :param val_losses: [description]
    :type val_losses: [type]
    :param best_epoch: [description]
    :type best_epoch: [type]
    """
    epoch_list = torch.arange(len(train_losses)) + 1

    ax.semilogy(epoch_list, train_losses, linestyle='solid', alpha=1, label='training loss')
    ax.semilogy(epoch_list, val_losses, linestyle='solid', alpha=0.7, label='validation loss')
    print(
        f'Best epoch ({best_epoch}): training loss {train_losses[best_epoch]:e}, validation loss {val_losses[best_epoch]:e}')
    ax.axvline(x=best_epoch, color='k', linestyle='dashed', label='best epoch')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    
    
def batched_model_inference(dataset, model, batch_size=5000):
      """Process dataset in batches to avoid memory issues.
      
      Args:
          dataset: A PyTorch dataset object with __getitem__ and __len__ methods
          model: The neural network model to use for inference
          batch_size: Number of samples to process at once
          
      """
      all_preds = []
      
      with torch.inference_mode():
            for i in range(0, len(dataset), batch_size):
                  # Get batch data and run inference
                  batch_x, batch_y = default_collate_fn([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
                  all_preds.append(model(batch_x))
                  
      return torch.cat(all_preds)
  

def default_collate_fn(batch):
    """ Collate function.
    Handles both ((image, features), target) and (features, target) formats.
    """
    # Extract structure from first item
    is_nested = isinstance(batch[0][0], (tuple, list))
    
    if is_nested:
        images = torch.stack([item[0][0] for item in batch])
        features = torch.stack([item[0][1] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        
        return ((images, features), targets)
    else:
        return (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch])
        )
