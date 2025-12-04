import torch
import numpy as np


def reshape_for_broadcasting(source, target):
    """Reshape source tensor to be compatible with target tensor for broadcasting."""
    if isinstance(source, torch.Tensor):
        source_shape = list(source.shape)
    else:
        source_shape = list(np.array(source).shape)
    
    if isinstance(target, torch.Tensor):
        target_shape = list(target.shape)
    else:
        target_shape = list(np.array(target).shape)
    
    # If they're already compatible, return as-is
    if len(source_shape) == len(target_shape):
        return source
    
    # Add dimensions to match target
    while len(source_shape) < len(target_shape):
        source_shape = [1] + source_shape
    
    if isinstance(source, torch.Tensor):
        return source.reshape(source_shape)
    else:
        return np.reshape(source, source_shape)


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf, device='cpu'):
        """A normalizer that ensures that observations are approximately distributed according to
        a standard Normal distribution (i.e. have mean zero and variance one).

        Args:
            size (int): the size of the observation to be normalized
            eps (float): a small constant that avoids underflows
            default_clip_range (float): normalized observations are clipped to be in
                [-default_clip_range, default_clip_range]
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.device = torch.device(device)

        # Use PyTorch tensors instead of TensorFlow variables
        self.local_sum = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.local_sumsq = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.local_count = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Running statistics (no MPI synchronization needed)
        self.sum = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.sumsq = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.count = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Computed statistics
        self.mean = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.std = torch.ones(self.size, dtype=torch.float32, device=self.device)

    def update(self, v):
        """Update the normalizer with a batch of observations."""
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float().to(self.device)
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32, device=self.device)

        # Handle single observation
        if v.dim() == 1:
            v = v.unsqueeze(0)

        # Update local statistics
        self.local_sum += torch.sum(v, dim=0)
        self.local_sumsq += torch.sum(v ** 2, dim=0)
        self.local_count += v.shape[0]

    def normalize(self, v, clip_range=None):
        """Normalize observations using current statistics."""
        if isinstance(v, np.ndarray):
            v_tensor = torch.from_numpy(v).float().to(self.device)
            return_numpy = True
        else:
            v_tensor = v.float().to(self.device)
            return_numpy = False

        if clip_range is None:
            clip_range = self.default_clip_range

        # Normalize
        normalized = (v_tensor - self.mean) / self.std
        
        # Clip
        if clip_range != np.inf:
            normalized = torch.clamp(normalized, -clip_range, clip_range)

        if return_numpy:
            return normalized.cpu().numpy()
        else:
            return normalized

    def denormalize(self, v):
        """Denormalize observations."""
        if isinstance(v, np.ndarray):
            v_tensor = torch.from_numpy(v).float().to(self.device)
            return_numpy = True
        else:
            v_tensor = v.float().to(self.device)
            return_numpy = False

        denormalized = v_tensor * self.std + self.mean

        if return_numpy:
            return denormalized.cpu().numpy()
        else:
            return denormalized

    def synchronize(self, local_sum=None, local_sumsq=None, local_count=None, root=None):
        """Synchronize statistics (simplified - no MPI in PyTorch version)."""
        if local_sum is not None:
            self.local_sum = torch.from_numpy(local_sum).float().to(self.device)
        if local_sumsq is not None:
            self.local_sumsq = torch.from_numpy(local_sumsq).float().to(self.device)
        if local_count is not None:
            self.local_count = torch.tensor(local_count, dtype=torch.float32, device=self.device)

        # Update global statistics (no MPI averaging)
        self.sum = self.local_sum.clone()
        self.sumsq = self.local_sumsq.clone()
        self.count = self.local_count.clone()

        self.recompute_stats()

    def recompute_stats(self):
        """Recompute mean and std from current statistics."""
        # Avoid division by zero
        count = torch.max(self.count, torch.tensor(1.0, device=self.device))
        
        self.mean = self.sum / count
        variance = (self.sumsq / count) - (self.mean ** 2)
        
        # Ensure variance is positive
        variance = torch.clamp(variance, min=self.eps)
        self.std = torch.sqrt(variance)


class IdentityNormalizer:
    def __init__(self, size, std=1., device='cpu'):
        """A normalizer that does nothing (identity transform)."""
        self.size = size
        self.std_val = std
        self.device = torch.device(device)

    def update(self, x):
        """No-op for identity normalizer."""
        pass

    def normalize(self, x, clip_range=None):
        """Return input unchanged."""
        return x

    def denormalize(self, x):
        """Return input unchanged."""
        return x

    def synchronize(self):
        """No-op for identity normalizer."""
        pass

    def recompute_stats(self):
        """No-op for identity normalizer."""
        pass