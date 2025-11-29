import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.util import store_args


def mlp(input_dim, hidden_size, num_layers, output_dim, output_activation=None):
    layers = []
    last_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.ReLU())
        last_dim = hidden_size

    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    @store_args
    def __init__(
        self,
        dimo,
        dimg,
        dimu,
        max_u,
        o_stats,
        g_stats,
        hidden,
        layers,
        **kwargs
    ):
        """
        PyTorch implementation of the actor–critic networks.
        Args are identical to the original TF version.
        """

        super().__init__()

        input_pi_dim = dimo + dimg
        input_Q_dim = dimo + dimg + dimu

        # Actor network
        self.actor = mlp(
            input_dim=input_pi_dim,
            hidden_size=hidden,
            num_layers=layers,
            output_dim=dimu,
        )

        # Critic network
        self.critic = mlp(
            input_dim=input_Q_dim,
            hidden_size=hidden,
            num_layers=layers,
            output_dim=1,
        )

        # store scalars/normalizers
        self._max_u = float(max_u)
        # keep a float tensor template for scaling; move to correct device at runtime
        self.max_u_tensor = torch.tensor(self._max_u, dtype=torch.float32)
        self.o_stats = o_stats
        self.g_stats = g_stats

    # small helper: ensure `x` is a torch tensor on same device & dtype as `ref`
    def _ensure_tensor(self, x, ref_tensor):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            # numpy array or other -> convert
            return torch.as_tensor(x, dtype=ref_tensor.dtype, device=ref_tensor.device)

    # ------------------------------------------------------------------
    # Forward functions
    # ------------------------------------------------------------------

    def pi(self, o, g):
        """Compute actions from observation + goal.

        This method is defensive: o/g may be torch tensors or numpy arrays (depending on the normalizer).
        We coerce normalized outputs to torch tensors before concatenation.
        """
        # Normalize (normalizer might return numpy arrays or tensors)
        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)

        # Convert to torch tensors if needed; use `o` as reference for device/dtype
        o_norm = self._ensure_tensor(o_norm, o)
        g_norm = self._ensure_tensor(g_norm, o)

        inp = torch.cat([o_norm, g_norm], dim=-1)
        act = self.actor(inp)
        act = torch.tanh(act) * self.max_u_tensor.to(inp.device)
        return act

    def Q(self, o, g, u):
        """Critic value for provided (o, g, u)."""

        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)

        o_norm = self._ensure_tensor(o_norm, o)
        g_norm = self._ensure_tensor(g_norm, o)
        # ensure action tensor
        u_t = self._ensure_tensor(u, o)

        u_scaled = u_t / self.max_u_tensor.to(o.device)
        inp = torch.cat([o_norm, g_norm, u_scaled], dim=-1)
        return self.critic(inp)

    def Q_pi(self, o, g):
        """Critic value evaluated at actor's actions."""
        pi_action = self.pi(o, g)

        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)

        o_norm = self._ensure_tensor(o_norm, o)
        g_norm = self._ensure_tensor(g_norm, o)
        pi_scaled = self._ensure_tensor(pi_action, o) / self.max_u_tensor.to(o.device)
        inp = torch.cat([o_norm, g_norm, pi_scaled], dim=-1)
        return self.critic(inp)
