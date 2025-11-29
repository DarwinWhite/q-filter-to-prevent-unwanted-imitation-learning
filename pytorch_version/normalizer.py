"""
A lightweight, NumPy-only normalizer suitable for single-process PyTorch training.

This file purposefully avoids importing mpi4py or tensorflow at module import time
so it does not trigger MPI initialization (which caused the PMPI_Init_thread abort).
It implements the same external API used by the rest of the codebase:
- Normalizer(size, eps=..., default_clip_range=..., sess=None)
- IdentityNormalizer(size, std=1.)
"""

import threading
import numpy as np
import warnings
import os

# Try to detect mpi4py only when explicitly asked; do NOT import it at module import time.
_MPI_AVAILABLE = False
_MPI = None
try:
    # Only attempt to import if user explicitly requested MPI via env var; otherwise skip.
    if os.environ.get("USE_MPI", "0") == "1":
        import mpi4py  # type: ignore
        from mpi4py import MPI  # type: ignore
        _MPI_AVAILABLE = True
        _MPI = MPI
except Exception:
    # Don't fail import — we deliberately avoid MPI in the default single-process case.
    _MPI_AVAILABLE = False
    _MPI = None


class Normalizer:
    """
    NumPy-based normalizer:
      - Collects local sums / sumsqs / counts via update()
      - recompute_stats() computes mean/std from accumulated local values
      - synchronize() is a best-effort collective that only runs if mpi4py was intentionally enabled
    """

    def __init__(self, size, eps=1e-2, default_clip_range=np.inf, sess=None):
        self.size = int(size)
        self.eps = float(eps)
        self.default_clip_range = default_clip_range

        # Local (accumulating) stats
        self.local_sum = np.zeros(self.size, dtype=np.float64)
        self.local_sumsq = np.zeros(self.size, dtype=np.float64)
        self.local_count = np.zeros(1, dtype=np.float64)

        # Global (exposed) stats (kept as float64 for numeric stability)
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.std = np.ones(self.size, dtype=np.float64)

        # Thread lock for update/recompute
        self.lock = threading.Lock()

    def __getstate__(self):
        """
        Make Normalizer pickle/deepcopy-safe:
        Remove the unpicklable lock from state and store a flag so we can
        recreate it when unpickling.
        """
        state = self.__dict__.copy()
        # Remove the lock (cannot be pickled)
        lock_present = 'lock' in state
        state.pop('lock', None)
        # keep a flag so __setstate__ knows to recreate it
        state['_had_lock'] = bool(lock_present)
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling/deepcopy and recreate the lock.
        """
        had_lock = state.pop('_had_lock', True)
        self.__dict__.update(state)
        # Recreate the lock (always create one for safety)
        self.lock = threading.Lock() if had_lock else threading.Lock()

    def update(self, v):
        """v should be shaped (-1, size)"""
        arr = np.asarray(v, dtype=np.float64).reshape(-1, self.size)
        with self.lock:
            self.local_sum += arr.sum(axis=0)
            self.local_sumsq += (arr ** 2).sum(axis=0)
            self.local_count[0] += arr.shape[0]

    def _mpi_average(self, x):
        """Perform an Allreduce if MPI was intentionally enabled; otherwise return local x."""
        if not _MPI_AVAILABLE:
            # single-process: just return x unchanged
            return x.copy()
        try:
            buf = np.zeros_like(x)
            _MPI.COMM_WORLD.Allreduce(x, buf, op=_MPI.SUM)
            buf = buf / _MPI.COMM_WORLD.Get_size()
            return buf
        except Exception as e:
            warnings.warn(f"MPI Allreduce failed in Normalizer._mpi_average: {e}")
            return x.copy()

    def synchronize(self, local_sum, local_sumsq, local_count, root=None):
        """If MPI enabled, average across ranks. Otherwise return inputs unchanged."""
        if _MPI_AVAILABLE:
            s = self._mpi_average(local_sum)
            ssq = self._mpi_average(local_sumsq)
            cnt = self._mpi_average(local_count)
            return s, ssq, cnt
        else:
            return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        """Apply accumulated local stats to compute mean/std and reset local accumulators."""
        with self.lock:
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            local_count = self.local_count.copy()
            # reset local accumulators
            self.local_sum[...] = 0.0
            self.local_sumsq[...] = 0.0
            self.local_count[...] = 0.0

        # If MPI is active, average across procs; otherwise just use local values.
        synced_sum, synced_sumsq, synced_count = self.synchronize(local_sum, local_sumsq, local_count)

        # Avoid division by zero
        cnt = float(synced_count[0]) if isinstance(synced_count, (list, np.ndarray)) else float(synced_count)
        if cnt <= 0:
            # nothing to update
            return

        mean = synced_sum / cnt
        var = (synced_sumsq / cnt) - (mean ** 2)
        var = np.maximum(var, self.eps ** 2)
        std = np.sqrt(var)

        # Update public stats
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)

    def normalize(self, v, clip_range=None):
        """Normalize a numpy array (returns numpy array). clip_range defaults to default_clip_range."""
        if clip_range is None:
            clip_range = self.default_clip_range
        arr = np.asarray(v, dtype=np.float64)
        # broadcast mean/std
        mean = self.mean
        std = self.std
        # Avoid divide by zero
        std_safe = np.where(std == 0, 1.0, std)
        normed = (arr - mean) / std_safe
        if clip_range is not None and np.isfinite(clip_range):
            normed = np.clip(normed, -clip_range, clip_range)
        return normed

    def denormalize(self, v):
        arr = np.asarray(v, dtype=np.float64)
        return self.mean + arr * self.std


class IdentityNormalizer:
    def __init__(self, size, std=1.):
        self.size = int(size)
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.std = float(std) * np.ones(self.size, dtype=np.float64)

    def __getstate__(self):
        """Make IdentityNormalizer pickle-safe (no lock present but keep consistent API)."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, x):
        return

    def normalize(self, x, clip_range=None):
        x = np.asarray(x, dtype=np.float64)
        return x / self.std

    def denormalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x * self.std

    def synchronize(self):
        return

    def recompute_stats(self):
        return
