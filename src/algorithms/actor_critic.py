import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.util import store_args


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 device='cpu', **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): placeholder dict for interface compatibility
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (Normalizer): normalizer for observations
            g_stats (Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.input_dim = dimo + dimg
        
        # Create actor and critic networks
        self.actor = ActorNetwork(self.input_dim, hidden, layers, dimu, max_u).to(self.device)
        self.critic = CriticNetwork(self.input_dim + dimu, hidden, layers).to(self.device)
        
    def get_actions(self, o, g):
        """Get actions from actor network"""
        # Normalize inputs
        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)
        
        # Concatenate observations and goals
        input_tensor = torch.cat([o_norm, g_norm], dim=1)
        
        with torch.no_grad():
            actions = self.actor(input_tensor)
        
        return actions
    
    def get_q_values(self, o, g, u):
        """Get Q-values from critic network"""
        # Normalize inputs
        o_norm = self.o_stats.normalize(o)
        g_norm = self.g_stats.normalize(g)
        
        # Concatenate observations, goals, and actions
        input_tensor = torch.cat([o_norm, g_norm, u / self.max_u], dim=1)
        
        with torch.no_grad():
            q_values = self.critic(input_tensor)
        
        return q_values


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden, layers, output_dim, max_u):
        """Actor network implementation in PyTorch.
        
        Args:
            input_dim (int): Input dimension (observations + goals)
            hidden (int): Hidden layer size
            layers (int): Number of hidden layers
            output_dim (int): Action dimension
            max_u (float): Maximum action magnitude
        """
        super(ActorNetwork, self).__init__()
        
        self.max_u = max_u
        
        # Build network layers
        layer_sizes = [input_dim] + [hidden] * layers + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x):
        """Forward pass through actor network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # Final layer with tanh activation, scaled by max_u
        x = self.max_u * torch.tanh(self.layers[-1](x))
        
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden, layers):
        """Critic network implementation in PyTorch.
        
        Args:
            input_dim (int): Input dimension (observations + goals + actions)
            hidden (int): Hidden layer size
            layers (int): Number of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        # Build network layers
        layer_sizes = [input_dim] + [hidden] * layers + [1]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x):
        """Forward pass through critic network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # Final layer (no activation for Q-values)
        x = self.layers[-1](x)
        
        return x


def create_actor_critic_networks(input_dims, hidden=256, layers=3, max_u=1.0, device='cpu'):
    """Helper function to create actor-critic networks with standard configuration.
    
    Args:
        input_dims (dict): Dictionary with 'o', 'g', 'u' dimensions
        hidden (int): Hidden layer size
        layers (int): Number of hidden layers
        max_u (float): Maximum action magnitude
        device (str): PyTorch device
        
    Returns:
        tuple: (actor, critic) networks
    """
    dimo, dimg, dimu = input_dims['o'], input_dims['g'], input_dims['u']
    input_dim = dimo + dimg
    
    actor = ActorNetwork(input_dim, hidden, layers, dimu, max_u).to(device)
    critic = CriticNetwork(input_dim + dimu, hidden, layers).to(device)
    
    return actor, critic