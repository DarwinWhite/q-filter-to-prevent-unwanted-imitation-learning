import os
import subprocess
import sys
import importlib
import inspect
import functools
import threading

import numpy as np
import torch
import torch.nn as nn


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}

    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))

    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)

    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        args = defaults.copy()

        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value

        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by 'pkg.module:fn_name'."""
    mod_name, fn_name = spec.split(":")
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)


def flatten_grads(var_list, grads):
    """
    Flattens PyTorch variable gradients into a single vector.
    var_list: iterable of tensors
    grads: iterable of gradient tensors
    """
    flat = [g.reshape(-1) for g in grads if g is not None]
    return torch.cat(flat, dim=0) if len(flat) > 0 else None


def build_mlp(
    input_dim,
    layer_sizes,
    activation=nn.ReLU,
    output_activation=None
):
    """
    Builds a simple MLP with PyTorch, similar to the TF-version of `nn()`.
    """
    layers = []
    last_dim = input_dim

    for size in layer_sizes[:-1]:
        layers.append(nn.Linear(last_dim, size))
        layers.append(activation())
        last_dim = size

    # final layer
    layers.append(nn.Linear(last_dim, layer_sizes[-1]))
    if output_activation:
        layers.append(output_activation())

    return nn.Sequential(*layers)


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()

    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with MPI workers."""
    if n <= 1:
        return "child"

    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")

        args = ["mpirun", "-np", str(n)] + extra_mpi_args + [sys.executable]
        args += sys.argv

        subprocess.check_call(args, env=env)
        return "parent"

    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Convert episode dict arrays from time-major to batch-major."""
    return {k: np.array(v).copy().swapaxes(0, 1) for k, v in episode.items()}


def transitions_in_episode_batch(episode_batch):
    """Return number of transitions in a batch-major episode."""
    shape = episode_batch["u"].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """
    Reshapes PyTorch tensor 'source' to broadcast correctly with 'target'.
    """
    dim = target.dim()
    shape = ([1] * (dim - 1)) + [-1]
    return source.to(target.dtype).reshape(shape)
