import numpy as np
import torch
from src.algorithms.ddpg import DDPG


class DDPGMuJoCo:
    """
    MuJoCo adapter for the PyTorch goal-conditioned DDPG implementation.
    Converts flat state observations into the (o, ag, g) format required by DDPG,
    generating dummy goals when the environment does not have goal structure.
    """

    def __init__(self, input_dims, device="cpu", **kwargs):
        """
        Args:
            input_dims: dict with 'o', 'u', 'g'
            device: torch device for tensors
            **kwargs: passed into the underlying DDPG constructor
        """

        # ensure minimal goal size
        if input_dims["g"] == 0:
            input_dims = input_dims.copy()
            input_dims["g"] = 1

        self.original_input_dims = input_dims.copy()
        self.has_goals = input_dims["g"] > 1
        self.device = torch.device(device)

        # instantiate the PyTorch DDPG
        self.ddpg = DDPG(input_dims=input_dims, device=device, **kwargs)

    def _create_dummy_goals(self, batch_size):
        """Return dummy goal tensors on the correct device."""
        g_dim = self.original_input_dims["g"] if self.has_goals else 1
        return torch.zeros((batch_size, g_dim), dtype=torch.float32, device=self.device)

    def _to_tensor(self, x):
        """Convert NumPy → Torch if necessary."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _preprocess_mujoco_inputs(self, o, ag=None, g=None):
        """
        Convert flat MuJoCo observations into tensors expected by PyTorch DDPG.
        """
        # convert obs
        o = self._to_tensor(o)
        if o.ndim == 1:
            o = o.unsqueeze(0)

        batch_size = o.shape[0]

        # if no real goals → dummy goals
        if not self.has_goals:
            ag = self._create_dummy_goals(batch_size)
            g = self._create_dummy_goals(batch_size)
        else:
            ag = self._to_tensor(ag)
            g = self._to_tensor(g)

        return o, ag, g

    def get_actions(self, o, ag=None, g=None, **kwargs):
        o, ag, g = self._preprocess_mujoco_inputs(o, ag, g)
        return self.ddpg.get_actions(o, ag, g, **kwargs)

    def store_episode(self, episode):
        """
        MuJoCo RolloutWorker already formats episodes into:
        { 'o':..., 'ag':..., 'g':..., 'u':... }
        But ensure everything is converted to torch.
        """
        torch_episode = {
            k: self._to_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in episode.items()
        }
        return self.ddpg.store_episode(torch_episode)

    def train(self):
        return self.ddpg.train()

    def update_target_net(self):
        return self.ddpg.update_target_net()

    def logs(self):
        return self.ddpg.logs()

    def save_policy(self, path):
        return self.ddpg.save_policy(path)

    def load_policy(self, path):
        return self.ddpg.load_policy(path)

    def initDemoBuffer(self, demo_file):
        return self.ddpg.initDemoBuffer(demo_file)

    def __getattr__(self, name):
        """Fallback delegation to underlying DDPG implementation."""
        return getattr(self.ddpg, name)
