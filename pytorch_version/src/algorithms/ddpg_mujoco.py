from src.algorithms.ddpg import DDPG


class DDPGMuJoCo:
    """
    MuJoCo adapter for the PyTorch goal-conditioned DDPG implementation.
    This wrapper allows the existing DDPG code to work with flat state observations
    by creating dummy goals and adapting the interface.
    """
    
    def __init__(self, input_dims, **kwargs):
        """
        Initialize DDPG adapter for MuJoCo environments.
        
        Args:
            input_dims: Dictionary with 'o' (observations), 'u' (actions), 'g' (goals=0 for MuJoCo)
            **kwargs: All other DDPG parameters
        """
        # Ensure we have minimal goal dimension for compatibility
        if input_dims['g'] == 0:
            input_dims = input_dims.copy()
            input_dims['g'] = 1  # Minimal goal dimension
            
        self.original_input_dims = input_dims.copy()
        self.has_goals = input_dims['g'] > 1  # True if real goals, False if dummy
        
        # Remove input_dims from kwargs if it exists to avoid duplicate argument
        if 'input_dims' in kwargs:
            kwargs.pop('input_dims')
        
        # Initialize the PyTorch DDPG with potentially modified dims
        self.ddpg = DDPG(input_dims=input_dims, **kwargs)
        
    def _create_dummy_goals(self, batch_size):
        """Create dummy goal arrays for compatibility with goal-conditioned DDPG"""
        import numpy as np
        return np.zeros((batch_size, self.ddpg.dimg))
    
    def _preprocess_mujoco_inputs(self, o, ag=None, g=None):
        """Preprocess MuJoCo inputs to be compatible with goal-conditioned interface"""
        import numpy as np
        
        if ag is None:
            ag = o  # Use observation as achieved goal for compatibility
        if g is None:
            if len(o.shape) == 1:
                # Single observation - create single goal
                g = np.zeros(self.ddpg.dimg)
            else:
                # Batch of observations - create batch of goals
                g = self._create_dummy_goals(o.shape[0])
        
        return o, ag, g
    
    def get_actions(self, o, ag=None, g=None, **kwargs):
        """Get actions from the policy."""
        o, ag, g = self._preprocess_mujoco_inputs(o, ag, g)
        return self.ddpg.get_actions(o, ag, g, **kwargs)
    
    def store_episode(self, episode):
        """Store episode in replay buffer."""
        # Ensure episode has dummy goals if needed
        if 'g' not in episode:
            batch_size = episode['o'].shape[0] if len(episode['o'].shape) > 1 else 1
            episode['g'] = self._create_dummy_goals(batch_size)
        if 'ag' not in episode:
            episode['ag'] = episode['o']  # Use observation as achieved goal
            
        self.ddpg.store_episode(episode)
    
    def train(self):
        """Train the DDPG agent."""
        return self.ddpg.train()
    
    def update_target_net(self):
        """Update target networks."""
        return self.ddpg.update_target_net()
    
    def logs(self):
        """Get logging information."""
        return self.ddpg.logs()
    
    def save_policy(self, path):
        """Save the policy."""
        return self.ddpg.save_policy(path)
    
    def load_policy(self, path):
        """Load a saved policy."""
        return self.ddpg.load_policy(path)
    
    def initDemoBuffer(self, demo_file, update_stats=True):
        """Initialize demonstration buffer."""
        return self.ddpg.initDemoBuffer(demo_file, update_stats=update_stats)
    
    def __getstate__(self):
        """Handle pickling of the wrapper."""
        state = self.__dict__.copy()
        # The DDPG object should handle its own state
        return state
    
    def __setstate__(self, state):
        """Handle unpickling of the wrapper."""
        self.__dict__.update(state)
        
        # Recreate any necessary objects
        if hasattr(self, 'ddpg'):
            # DDPG should restore its own state
            pass
        else:
            # If DDPG is missing, we'll need to recreate it
            # This should not happen in normal usage
            print("Warning: DDPG object missing during unpickling")
    
    # Delegate other attributes to the wrapped DDPG instance
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped DDPG instance."""
        if hasattr(self.ddpg, name):
            return getattr(self.ddpg, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")