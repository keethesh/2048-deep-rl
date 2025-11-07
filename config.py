"""
FIXED Training Configuration - Copy this to config.py
This configuration will actually allow the AI to learn!
"""

# ============================================================================
# TRAINING MODE
# ============================================================================

# Fast mode: Moderate training for testing (1-2 hours CPU, 15-25 min GPU)
# Full mode: Complete training for best performance (3-5 hours CPU, 30-60 min GPU)
FAST_MODE = True

# ============================================================================
# TRAINING HYPERPARAMETERS - FIXED FOR ACTUAL LEARNING
# ============================================================================

if FAST_MODE:
    NUM_EPISODES = 1000         # Sufficient for learning
    BATCH_SIZE = 32             # ❗ SMALLER = more stable updates (was 128)
    LEARNING_RATE = 0.00005     # ❗ MUCH LOWER = stable convergence (was 0.0003)
    EPSILON_DECAY = 0.998       # ❗ SLOWER = explores longer (was 0.99)
    TARGET_UPDATE_FREQ = 5      # ❗ MORE FREQUENT = better sync (was 20)
    EVAL_FREQ = 100             # Less frequent evaluation
    EVAL_EPISODES = 3           # Faster evaluation
    SAVE_FREQ = 200             # Less frequent saves
    LOG_FREQ = 10               # Less console spam
    MEMORY_SIZE = 50000         # ❗ LARGER = more diversity (was 20000)
else:
    NUM_EPISODES = 5000         # Full training
    BATCH_SIZE = 64             # Standard batch size
    LEARNING_RATE = 0.00005     # ❗ More conservative (was 0.0001)
    EPSILON_DECAY = 0.997       # ❗ Slower decay (was 0.995)
    TARGET_UPDATE_FREQ = 10     # Frequent target updates
    EVAL_FREQ = 100             # Regular evaluation
    EVAL_EPISODES = 5           # Moderate evaluation
    SAVE_FREQ = 100             # Regular saves
    LOG_FREQ = 10               # Moderate logging
    MEMORY_SIZE = 50000         # Large replay buffer

# Common hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05              # ❗ HIGHER minimum (was 0.01) - keeps some exploration

# ============================================================================
# NETWORK ARCHITECTURE - FIXED FOR SUFFICIENT CAPACITY
# ============================================================================

if FAST_MODE:
    CONV_FILTERS = 128          # ❗ LARGER network (was 64) - can actually learn!
    FC_SIZE = 256               # ❗ LARGER layers (was 128) - more capacity
else:
    CONV_FILTERS = 128          # Larger, more powerful network
    FC_SIZE = 256               # Larger fully connected layers

# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================

# Mixed precision training (GPU only, requires CUDA)
USE_MIXED_PRECISION = True      # ~1.5x speedup on GPU

# Gradient clipping
MAX_GRAD_NORM = 10.0

# Prioritized Experience Replay
PER_ALPHA = 0.6                 # Priority exponent
PER_BETA_START = 0.4            # Importance sampling weight
PER_BETA_FRAMES = 100000        # Frames to anneal beta to 1.0

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# Auto-detect best device
import torch
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# ============================================================================
# PATHS
# ============================================================================

MODELS_DIR = "models"
RUNS_DIR = "runs"
CHECKPOINTS_PREFIX = "checkpoint_ep"
BEST_MODEL_PREFIX = "best_model"
FINAL_MODEL_NAME = "final_model.pth"

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

PRINT_SUMMARY = True            # Print training summary at start
SHOW_PROGRESS_BAR = False       # Use tqdm progress bar (requires tqdm)
VERBOSE = True                  # Detailed logging

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

DEBUG_MODE = False              # Extra validation and logging
PROFILE_CODE = False            # Profile with cProfile
DETERMINISTIC = False           # Deterministic training (slower)

if DETERMINISTIC:
    import numpy as np
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if USE_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# SUMMARY
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print(f"{'TRAINING CONFIGURATION':^70}")
    print("=" * 70)
    print(f"Mode: {'FAST (Testing)' if FAST_MODE else 'FULL (Production)'}")
    print(f"Device: {DEVICE.upper()}{' (CUDA)' if USE_CUDA else ' (CPU)'}")
    print("-" * 70)
    print(f"Episodes: {NUM_EPISODES:,}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE} (FIXED - was 0.0003)")
    print(f"Memory Size: {MEMORY_SIZE:,} (FIXED - was 20000)")
    print(f"Network Filters: {CONV_FILTERS} (FIXED - was 64)")
    print(f"Epsilon Decay: {EPSILON_DECAY} (FIXED - was 0.99)")
    print(f"Target Update Freq: {TARGET_UPDATE_FREQ} (FIXED - was 20)")
    print("-" * 70)
    print(f"Eval Frequency: Every {EVAL_FREQ} episodes ({EVAL_EPISODES} episodes)")
    print(f"Save Frequency: Every {SAVE_FREQ} episodes")
    print(f"Log Frequency: Every {LOG_FREQ} episodes")
    print("-" * 70)
    print(f"Mixed Precision: {'Enabled' if USE_MIXED_PRECISION and USE_CUDA else 'Disabled'}")
    print(f"Gradient Clipping: {MAX_GRAD_NORM}")
    print("=" * 70)
    print()
    print("CHANGES FROM ORIGINAL CONFIG:")
    print("  - Batch size: 128 -> 32 (more stable updates)")
    print("  - Learning rate: 0.0003 -> 0.00005 (5x more stable)")
    print("  - Epsilon decay: 0.99 -> 0.998 (explores 5x longer)")
    print("  - Target update: 20 -> 5 (4x more frequent sync)")
    print("  - Memory: 20k -> 50k (2.5x more diversity)")
    print("  - Network: 64 -> 128 filters (2x capacity)")
    print("  - FC size: 128 -> 256 (2x capacity)")
    print("=" * 70)
    print()

# Estimated training time
if FAST_MODE:
    if USE_CUDA:
        print(f"Estimated training time: 15-25 minutes (GPU)")
    else:
        print(f"Estimated training time: 1.5-2.5 hours (CPU)")
else:
    if USE_CUDA:
        print(f"Estimated training time: 30-60 minutes (GPU)")
    else:
        print(f"Estimated training time: 3-5 hours (CPU)")
print()
