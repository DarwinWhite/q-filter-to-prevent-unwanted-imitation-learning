import numpy as np
from src.algorithms.ddpg import DDPG


class DDPGMuJoCo:
    """
    MuJoCo adapter for the original goal-conditioned DDPG implementation.
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
            input_dims['g'] = 1  # Minimum dimension for goal normalizers
            
        self.original_input_dims = input_dims.copy()
        self.has_goals = input_dims['g'] > 1  # True if real goals, False if dummy
        
        # Initialize the original DDPG with potentially modified dims
        self.ddpg = DDPG(input_dims=input_dims, **kwargs)
        
    def _create_dummy_goals(self, batch_size):
        """Create dummy goal arrays for compatibility with goal-conditioned DDPG"""
        return np.zeros((batch_size, self.original_input_dims['g'] if self.has_goals else 1), dtype=np.float32)
    
    def _preprocess_mujoco_inputs(self, o, ag=None, g=None):
        """
        Preprocess inputs for MuJoCo environments.
        Convert flat state observations to goal-conditioned format expected by DDPG.
        
        Args:
            o: Observations (flat state for MuJoCo)
            ag: Achieved goals (unused for MuJoCo, will be created as dummy)
            g: Desired goals (unused for MuJoCo, will be created as dummy)
        """
        # Ensure o is properly shaped
        if o.ndim == 1:
            o = o.reshape(1, -1)
        batch_size = o.shape[0]
        
        # Create dummy goals if not provided or if we don't have real goals
        if not self.has_goals:
            ag = self._create_dummy_goals(batch_size)
            g = self._create_dummy_goals(batch_size)
        
        return o, ag, g
    
    def get_actions(self, o, ag=None, g=None, **kwargs):
        """
        Get actions from the DDPG policy.
        Adapts flat state inputs to goal-conditioned format.
        
        Args:
            o: Observations (flat state for MuJoCo)
            ag: Achieved goals (will be created as dummy if None)
            g: Desired goals (will be created as dummy if None)
            **kwargs: Additional arguments for DDPG.get_actions()
        """
        o, ag, g = self._preprocess_mujoco_inputs(o, ag, g)
        return self.ddpg.get_actions(o, ag, g, **kwargs)
    
    def store_episode(self, episode):
        """
        Store an episode in the replay buffer.
        Ensure episode has correct format for goal-conditioned DDPG.
        """
        # The episode should already be properly formatted by RolloutWorkerMuJoCo
        # with dummy goal arrays for compatibility
        return self.ddpg.store_episode(episode)
    
    def train(self):
        """Train the DDPG agent"""
        return self.ddpg.train()
    
    def update_target_net(self):
        """Update target network"""
        return self.ddpg.update_target_net()
    
    def logs(self):
        """Get training logs"""
        return self.ddpg.logs()
    
    def save_policy(self, path):
        """Save policy"""
        return self.ddpg.save_policy(path)
    
    def load_policy(self, path):
        """Load policy"""
        return self.ddpg.load_policy(path)
    
    def initDemoBuffer(self, demo_file):
        """Initialize demo buffer for behavior cloning"""
        return self.ddpg.initDemoBuffer(demo_file)
    
    def __getstate__(self):
        """Handle pickling by delegating to the wrapped DDPG instance"""
        return {
            'original_input_dims': self.original_input_dims,
            'has_goals': self.has_goals,
            'ddpg': self.ddpg
        }
    
    def __setstate__(self, state):
        """Handle unpickling by restoring the wrapped DDPG instance"""
        # Check if this is our new wrapper format
        if isinstance(state, dict) and 'original_input_dims' in state and 'ddpg' in state:
            # New format with wrapper state
            self.original_input_dims = state['original_input_dims']
            self.has_goals = state['has_goals']
            self.ddpg = state['ddpg']
        elif hasattr(state, '__dict__') and hasattr(state, 'ddpg'):
            # This is already a DDPGMuJoCo object, just copy its attributes
            self.__dict__.update(state.__dict__)
        else:
            # Old format - state is the serialized DDPG state
            from src.algorithms.ddpg import DDPG
            
            # Create a new DDPG instance and restore its state
            self.ddpg = DDPG.__new__(DDPG)  # Create without calling __init__
            self.ddpg.__setstate__(state)   # Let DDPG handle its own state restoration
            
            # Try to infer the original dimensions from the DDPG instance
            try:
                dims = self.ddpg.input_dims
                self.original_input_dims = dims.copy()
                self.has_goals = dims['g'] > 1
            except Exception as e:
                # Fallback values for HalfCheetah
                self.original_input_dims = {'o': 17, 'g': 0, 'u': 6}
                self.has_goals = False
    
    # Delegate other attributes to the wrapped DDPG instance
    def __getattr__(self, name):
        """Delegate any other method calls to the wrapped DDPG instance"""
        if name == 'ddpg':
            # Prevent recursion when trying to access ddpg itself
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.ddpg, name)