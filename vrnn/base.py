"""
Basic PyTorch modules taken from MLPRUM
"""

import torch
from torch import nn
import copy


class BaseModule(nn.Module):
    """Represents a `Base Module` that contains the basic functionality of an artificial neural network (ANN).

    All modules should inherit from :class:`.models.BaseModule` and override :meth:`vrnn.models.BaseModule.forward`.
    The Base Module itself inherits from :class:`torch.nn.Module`. See the `PyTorch` documentation for further information.
    """

    def __init__(self):
        """Constructor of the class. Initialize the Base Module.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`vrnn.models.BaseModule` should not be instantiated itself, but only its subclasses.
        """
        super().__init__()
        self.device = 'cpu'
        self.dtype = torch.float32

    def forward(*args):
        """Forward propagation of the ANN. Subclasses must override this method.

        :raises NotImplementedError: If the method is not overriden by a subclass.
        """
        raise NotImplementedError('subclasses must override forward()!')

    def training_step(self, dataloader, loss_fn, optimizer):
        """Single training step that performs the forward propagation of an entire batch,
        the training loss calculation and a subsequent optimization step.

        A training epoch must contain one call to this method.

        Example:
            >>> train_loss = module.training_step(train_loader, loss_fn, optimizer)

        :param dataloader: Dataloader with training data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for the model training
        :type loss_fn: method
        :param optimizer: Optimizer for model training
        :type optimizer: :class:`torch.optim.Optimizer`
        :return: Training loss
        :rtype: float
        """
        self.train()  # enable training mode
        cumulative_loss = 0
        samples = 0
        for x, y in dataloader:
            if isinstance(x, tuple):
                x = tuple(elem.to(self.device, dtype=self.dtype) for elem in x)
                size = x[0].size(0)
            else:
                x = x.to(self.device, dtype=self.dtype)
                size = x.size(0)
                
            # Loss calculation
            optimizer.zero_grad()
            y_pred = self(x)
            loss = loss_fn(y_pred, y)
            cumulative_loss += loss.item() * size
            samples += size

            # Backpropagation
            loss.backward()
            optimizer.step()

        average_loss = cumulative_loss / samples
        return average_loss

    def loss_calculation(self, dataloader, loss_fns):
        """Perform the forward propagation of an entire batch from a given `dataloader`
        and the subsequent loss calculation for one or multiple loss functions in `loss_fns`.

        Example:
            >>> val_loss = module.loss_calculation(val_loader, loss_fn)

        :param dataloader: Dataloader with validation data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for model training
        :type loss_fn: method or list of methods
        :return: Validation loss
        :rtype: float
        """
        self.eval()  # disable training mode
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]
        cumulative_loss = torch.zeros(len(loss_fns))
        samples = 0
        with torch.inference_mode():  # disable gradient calculation
            for x, y in dataloader:
                if isinstance(x, tuple):
                    x = tuple(elem.to(self.device, dtype=self.dtype) for elem in x)
                    size = x[0].size(0)
                else:
                    x = x.to(self.device, dtype=self.dtype)
                    size = x.size(0)
                y_pred = self(x)
                samples += size
                for i, loss_fn in enumerate(loss_fns):
                    loss = loss_fn(y_pred, y)
                    cumulative_loss[i] += loss.item() * size
        average_loss = cumulative_loss / samples
        if torch.numel(average_loss) == 1:
            average_loss = average_loss[0]
        return average_loss

    def parameter_count(self):
        """Get the number of learnable parameters, that are contained in the model.

        :return: Number of learnable parameters
        :rtype: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def reduce(cls, loss, reduction='mean'):
        """Perform a reduction step over all datasets to transform a loss function to a cost function.

        A loss function is evaluated element-wise for a dataset.
        However, a cost function should return a single value for the dataset.
        Typically, `mean` reduction is used.

        :param loss: Tensor that contains the element-wise loss for a dataset
        :type loss: :class:`torch.Tensor`
        :param reduction: ('mean'|'sum'), defaults to 'mean'
        :type reduction: str, optional
        :return: Reduced loss
        :rtype: float
        """
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

    @classmethod
    def unsqueeze(cls, output, target):
        """Ensure that the tensors :code:`output` and :code:`target` have a shape of the form :code:`(N, features)`.

        When a loss function is called with a single data point, the tensor shape is :code:`(features)` and hence does not fit.
        This method expands the dimensions if needed.

        :param output: Model output
        :type output: :class:`torch.Tensor`
        :param target: Target data
        :type target: :class:`torch.Tensor`
        :return: Tuple (output, target)
        :rtype: tuple
        """
        if output.dim() == 1:
            output = torch.unsqueeze(output, 0)
        if target.dim() == 1:
            target = torch.unsqueeze(target, 0)
        return output, target

    def to(self, device, dtype=None, *args, **kwargs):
        """Transfers a model to another device, e.g. to a GPU.

        This method overrides the PyTorch built-in method :code:`model.to(...)`.

        Example:
            >>> # Transfer the model to a GPU
            >>> module.to('cuda:0')

        :param device: Identifier of the device, e.g. :code:`'cpu'`, :code:`'cuda:0'`, :code:`'cuda:1'`, ...
        :type device: str
        :return: The model itself
        :rtype: :class:`vrnn.models.BaseModule`
        """
        self.device = device

        if dtype is None:
            return super().to(device=device, *args, **kwargs)
        else:
            self.dtype = dtype
            return super().to(device=device, dtype=dtype, *args, **kwargs)

    @property
    def gpu(self):
        """Property, that indicates, whether the model is on a GPU, i.e., not on the CPU.

        :return: True, iff the module is not on the CPU
        :rtype: bool
        """
        return self.device != 'cpu'