"""
Voigt-Reuss PyTorch modules
"""
from typing import Collection

import torch
import torch.nn as nn
import torch.nn.functional as F
from vrnn.base import BaseModule
from vrnn.tensortools import pack_sym

class HillModule(BaseModule):
    """
    Hill module.
    """
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, bounds_fn) -> torch.Tensor:
        lower_bound, upper_bound = bounds_fn(x)
        kappa_pred = 0.5 * (lower_bound + upper_bound)
        kappa_dof = pack_sym(kappa_pred, dim=self.dim)
        return kappa_dof

class VanillaModule(BaseModule):
    def __init__(self, ann_module: torch.nn.Module, dim: int = 3):
        super().__init__()
        self.ann_module = ann_module
        self.dim = dim

    def forward(self, x):
        return self.ann_module(x)
        

class MixedActivationLayer(torch.nn.Module):
    def __init__(self, activation_fns=None):
        super().__init__()
        if activation_fns is None:
            activation_fns = [torch.nn.ReLU()]
        if not isinstance(activation_fns, Collection):
            activation_fns = [activation_fns]
        self.activation_fns = activation_fns
        self.n_act = len(activation_fns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_size = x.shape[-1]
        
        # Pre-compute all indices at once
        indices = []
        start = 0
        base_neurons = hidden_size // self.n_act
        extra_neurons = hidden_size % self.n_act
        
        for i in range(self.n_act):
            neurons = base_neurons + (1 if i < extra_neurons else 0)
            indices.append((start, start + neurons))
            start += neurons
        
        # Process in a single pass without creating a zero tensor first
        if self.n_act == 1:
            return self.activation_fns[0](x)
            
        # For multiple activations, process and write results directly
        y = torch.empty_like(x)
        for (start, end), act_fn in zip(indices, self.activation_fns):
            y[..., start:end] = act_fn(x[..., start:end])
        
        return y

class MixedActivationMLP(nn.Module):
    """
    Multi-layer perceptron with mixed activation functions between layers.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fns=None, 
                 output_activation=nn.Sigmoid(), use_batch_norm=True):
        super(MixedActivationMLP, self).__init__()
        
        if activation_fns is None:
            activation_fns = [nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(MixedActivationLayer(activation_fns))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class PeriodicConv(nn.Module):
    """
    Generic periodic convolution layer in 2D and 3D.
    """
    def __init__(self, in_channels, out_channels, kernel_size, n_dim=2, stride=1, dilation=1, groups=1, bias=True):
        super(PeriodicConv, self).__init__()
        self.n_dim = n_dim
        # Ensure kernel_size is a tuple of length n_dim.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dim

        if n_dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=0, dilation=dilation,
                                  groups=groups, bias=bias)
            # F.pad expects (left, right, top, bottom)
            pad = (kernel_size[1] // 2, kernel_size[1] // 2,
                   kernel_size[0] // 2, kernel_size[0] // 2)
        elif n_dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=0, dilation=dilation,
                                  groups=groups, bias=bias)
            # For 3D: pad is (w_left, w_right, h_top, h_bottom, d_front, d_back)
            pad = (kernel_size[2] // 2, kernel_size[2] // 2,
                   kernel_size[1] // 2, kernel_size[1] // 2,
                   kernel_size[0] // 2, kernel_size[0] // 2)
        else:
            raise ValueError("n_dim must be 2 or 3")
        self.pad = pad

    def forward(self, x):
        x = F.pad(x, self.pad, mode='circular')
        return self.conv(x)

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection and pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, n_dim=2, pool_kernel=2, pool_stride=2):
        super(ResidualBlock, self).__init__()
        self.n_dim = n_dim
        self.conv = PeriodicConv(in_channels, out_channels, kernel_size, n_dim=n_dim)
        # Choose appropriate batch norm and pooling layers.
        bn_layer = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        pool_layer = nn.AvgPool2d if n_dim == 2 else nn.AvgPool3d
        
        self.bn = bn_layer(out_channels)
        self.relu = nn.ReLU()
        self.pool = pool_layer(kernel_size=pool_kernel, stride=pool_stride)
        
        # Use a shortcut (1x1 or 1x1x1 convolution) if channels differ.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                PeriodicConv(in_channels, out_channels, kernel_size=1, n_dim=n_dim),
                bn_layer(out_channels),
                pool_layer(kernel_size=pool_kernel, stride=pool_stride)
            )
        else:
            self.shortcut = pool_layer(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        shortcut = self.shortcut(x)
        return self.relu(out + shortcut)

class CNNPlusScalars(nn.Module):
    def __init__(self, scalar_input_dim=3, fc_hidden_dims=[256, 128, 64, 32, 16],
                 out_dim=3, conv_channels=[16, 32, 64, 128], kernel_sizes=3,
                 initial_pool_size=4, img_size=400, n_dim=2, output_activation=nn.Sigmoid()):
        """
        Parameters:
            scalar_input_dim: Number of scalar features.
            fc_hidden_dims: Hidden dimensions for the FC part.
            out_dim: Output dimension.
            conv_channels: List of channel numbers for successive convolutional blocks.
            kernel_sizes: An int or list of ints/tuples (for each block).
            initial_pool_size: Pooling factor applied initially.
            img_size: Expected spatial size (assumed square or cubic) of input images.
            n_dim: 2 for 2D images, 3 for 3D images.
        """
        super(CNNPlusScalars, self).__init__()
        self.n_dim = n_dim
        pool_layer = nn.AvgPool2d if n_dim == 2 else nn.AvgPool3d
        
        # Initial pooling.
        self.initial_pool = pool_layer(kernel_size=initial_pool_size, stride=initial_pool_size)
        
        # Ensure kernel_sizes is a list matching conv_channels.
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        elif len(kernel_sizes) != len(conv_channels):
            raise ValueError("kernel_sizes must have the same length as conv_channels")
        
        # Build CNN branch (will process images)
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(conv_channels):
            self.conv_layers.append(
                ResidualBlock(in_channels, out_channels, kernel_size=kernel_sizes[i], n_dim=n_dim)
            )
            in_channels = out_channels
        
        # Compute final spatial dimensions after pooling.
        new_size = img_size // initial_pool_size
        # Each ResidualBlock's pooling halves each spatial dimension.
        for _ in conv_channels:
            new_size //= 2
        # Final feature size: product of channels and spatial dims.
        cnn_output_size = conv_channels[-1] * (new_size ** n_dim)
        fc_input_dim = cnn_output_size + scalar_input_dim
        
        # MixedActivationMLP for FC layers (will process combined features)
        self.fc = MixedActivationMLP(
            input_dim=fc_input_dim,
            hidden_dims=fc_hidden_dims,
            output_dim=out_dim,
            activation_fns=[nn.SELU(), nn.Tanh(), nn.Sigmoid(), nn.Identity()],
            output_activation=output_activation,
            use_batch_norm=True
        )
    
    def forward(self, x):
        if not isinstance(x, (tuple, list)):
            raise ValueError("Expected input as tuple (image, scalars)")
        image, scalars = x

        x_img = self.initial_pool(image)
        for layer in self.conv_layers:
            x_img = layer(x_img)
        x_img = x_img.view(x_img.size(0), -1)
        
        combined = torch.cat([x_img, scalars], dim=1)
        
        return self.fc(combined)
    