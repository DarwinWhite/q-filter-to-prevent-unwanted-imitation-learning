"""
TensorFlow compatibility helper for running TF 1.x code on TF 2.x
Add 'import tf_compat' at the top of any file that imports tensorflow
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    # Make TF 1.x available as 'tensorflow' for backward compatibility
    import sys
    sys.modules['tensorflow'] = tf
    
    print("✓ TensorFlow 2.x loaded in v1 compatibility mode")
    
except ImportError:
    # If TF 1.x is actually installed, just import normally
    import tensorflow as tf
    print("✓ TensorFlow 1.x loaded normally")

# Additional compatibility fixes
if hasattr(tf, 'Session'):
    # TF 1.x style - should work as is
    pass
else:
    # If needed, add more compatibility shims here
    pass