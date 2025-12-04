import torch
import torch.nn as nn
import numpy as np
import functools
import importlib
import inspect


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults.update(dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults)))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get the names and values of the arguments.
        positional_args = positional_args[1:]
        named_args = {}
        named_args.update(defaults)
        named_args.update(dict(zip(arg_names, positional_args)))
        named_args.update(keyword_args)
        # Set the attributes on self.
        for key, value in named_args.items():
            setattr(self, key, value)
        return method(self, **named_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(parameters, grads):
    """Flattens parameters and their gradients for PyTorch.
    """
    return torch.cat([grad.reshape(-1) for grad in grads])


def nn_pytorch(input_tensor, layer_sizes, activation=nn.ReLU, final_activation=None):
    """Creates a simple neural network in PyTorch.
    
    Args:
        input_tensor: Input tensor
        layer_sizes: List of layer sizes (not including input)
        activation: Activation function for hidden layers
        final_activation: Activation for final layer (None for no activation)
    
    Returns:
        Output tensor
    """
    layers = []
    
    # Get input size
    if len(input_tensor.shape) == 1:
        input_size = input_tensor.shape[0]
    else:
        input_size = input_tensor.shape[-1]
    
    # Build layers
    prev_size = input_size
    for i, size in enumerate(layer_sizes):
        layers.append(nn.Linear(prev_size, size))
        
        # Add activation except for last layer (unless specified)
        if i < len(layer_sizes) - 1:
            if activation is not None:
                layers.append(activation())
        else:
            if final_activation is not None:
                layers.append(final_activation())
                
        prev_size = size
    
    # Create sequential model
    model = nn.Sequential(*layers)
    return model(input_tensor)


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # Only swap axes if array has at least 2 dimensions
        if val.ndim >= 2:
            # Swap first and second dimensions
            episode_batch[key] = val.swapaxes(0, 1)
        else:
            # Keep 1D arrays as-is
            episode_batch[key] = val

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in this episode batch."""
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshape source to be broadcastable with target."""
    if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
        # PyTorch tensor reshaping
        dim_diff = target.dim() - source.dim()
        if dim_diff > 0:
            for _ in range(dim_diff):
                source = source.unsqueeze(-1)
        return source
    else:
        # NumPy array reshaping (original behavior)
        dim = len(target.shape)
        shape = source.shape
        return source.reshape(shape + (1,) * (dim - len(shape)))


def to_tensor(x, device='cpu', dtype=torch.float32):
    """Convert input to PyTorch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        return torch.tensor(x, device=device, dtype=dtype)


def to_numpy(x):
    """Convert PyTorch tensor to NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def polyak_update(target_params, source_params, polyak):
    """Update target network parameters using Polyak averaging.
    
    Args:
        target_params: Target network parameters
        source_params: Source network parameters  
        polyak: Polyak averaging coefficient
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            target_param.copy_(polyak * target_param + (1.0 - polyak) * source_param)


def hard_update(target_params, source_params):
    """Copy source network parameters to target network.
    
    Args:
        target_params: Target network parameters
        source_params: Source network parameters
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            target_param.copy_(source_param)


def set_requires_grad(network, requires_grad=True):
    """Set requires_grad for all parameters in a network.
    
    Args:
        network: PyTorch network
        requires_grad: Whether to require gradients
    """
    for param in network.parameters():
        param.requires_grad = requires_grad


# Simplified logging for PyTorch version (replaces MPI functionality)
def simple_average(value):
    """Simple averaging (no MPI in PyTorch version)."""
    if isinstance(value, (list, tuple)):
        return np.mean(value)
    else:
        return float(value)