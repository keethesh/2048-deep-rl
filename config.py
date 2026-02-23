"""
FIXED Training Configuration - Copy this to config.py
This configuration will actually allow the AI to learn!
"""

import os
import torch

# ============================================================================
# TRAINING MODE
# ============================================================================

# Fast mode: Moderate training for testing (1-2 hours CPU, 15-25 min GPU)
# Full mode: Complete training for best performance (3-5 hours CPU, 30-60 min GPU)
FAST_MODE = True
IS_COLAB = "COLAB_GPU" in os.environ
COLAB_SPEED_PROFILE = IS_COLAB and torch.cuda.is_available()

# ============================================================================
# TRAINING HYPERPARAMETERS - FIXED FOR ACTUAL LEARNING
# ============================================================================

if FAST_MODE:
    NUM_EPISODES = 4000 if COLAB_SPEED_PROFILE else 5000
    BATCH_SIZE = 128 if COLAB_SPEED_PROFILE else 64
    LEARNING_RATE = 0.0003      # Higher with scheduler
    EPSILON_DECAY = 0.9997     # Even slower exploration decay
    TARGET_UPDATE_FREQ = None   # Use soft updates instead
    EVAL_FREQ = 250 if COLAB_SPEED_PROFILE else 100
    EVAL_EPISODES = 3 if COLAB_SPEED_PROFILE else 10
    SAVE_FREQ = 500 if COLAB_SPEED_PROFILE else 200
    LOG_FREQ = 25 if COLAB_SPEED_PROFILE else 20
    MEMORY_SIZE = 100000        # DOUBLED: More diverse experiences (was 50000)
    TRAIN_FREQ = 8 if COLAB_SPEED_PROFILE else 4
    TAU = 0.005                 # Soft target update coefficient
    N_STEPS = 3                 # Multi-step returns
else:
    NUM_EPISODES = 5000         # Full training
    BATCH_SIZE = 128 if COLAB_SPEED_PROFILE else 64
    LEARNING_RATE = 0.0003      # Higher with scheduler
    EPSILON_DECAY = 0.9997     # Even slower exploration decay
    TARGET_UPDATE_FREQ = None   # Use soft updates instead
    EVAL_FREQ = 200 if COLAB_SPEED_PROFILE else 100
    EVAL_EPISODES = 3 if COLAB_SPEED_PROFILE else 5
    SAVE_FREQ = 500 if COLAB_SPEED_PROFILE else 100
    LOG_FREQ = 25 if COLAB_SPEED_PROFILE else 10
    MEMORY_SIZE = 50000         # Large replay buffer
    TRAIN_FREQ = 8 if COLAB_SPEED_PROFILE else 4
    TAU = 0.005                 # Soft target update coefficient
    N_STEPS = 3                 # Multi-step returns

# Common hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01              # LOWER: Allow more exploitation once strategies are learned (was 0.05)

# ============================================================================
# NETWORK ARCHITECTURE - FIXED FOR SUFFICIENT CAPACITY
# ============================================================================

if FAST_MODE:
    if COLAB_SPEED_PROFILE:
        CONV_FILTERS = 128      # Better throughput on Colab GPUs
        FC_SIZE = 256
    else:
        CONV_FILTERS = 256      # Higher-capacity default
        FC_SIZE = 512
else:
    if COLAB_SPEED_PROFILE:
        CONV_FILTERS = 128
        FC_SIZE = 256
    else:
        CONV_FILTERS = 256
        FC_SIZE = 512

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

# Learning Rate Scheduler
LR_SCHEDULER_T_0 = 1000         # Restart period for cosine annealing
LR_SCHEDULER_T_MULT = 2         # Period multiplier after each restart
LR_SCHEDULER_ETA_MIN = 1e-6     # Minimum learning rate

# Runtime acceleration options
TORCH_COMPILE = True            # Use torch.compile when available (GPU only in trainer)
TRACK_REWARD_COMPONENTS = False # Disable detailed reward breakdown during train for speed
EVAL_DEBUG_ON_FIRST = False     # Avoid expensive verbose eval prints

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# Auto-detect best device
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
    print(f"Colab speed profile: {'ON' if COLAB_SPEED_PROFILE else 'OFF'}")
    print("-" * 70)
    print(f"Episodes: {NUM_EPISODES:,}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Memory Size: {MEMORY_SIZE:,}")
    print(f"Network Filters: {CONV_FILTERS}")
    print(f"Epsilon Decay: {EPSILON_DECAY}")
    print("-" * 70)
    print(f"Eval Frequency: Every {EVAL_FREQ} episodes ({EVAL_EPISODES} episodes)")
    print(f"Save Frequency: Every {SAVE_FREQ} episodes")
    print(f"Log Frequency: Every {LOG_FREQ} episodes")
    print("-" * 70)
    print(f"Mixed Precision: {'Enabled' if USE_MIXED_PRECISION and USE_CUDA else 'Disabled'}")
    print(f"Gradient Clipping: {MAX_GRAD_NORM}")
    print(f"Training Frequency: Every {TRAIN_FREQ} steps")
    print(f"Soft Target Updates: tau={TAU}")
    print(f"N-Step Returns: {N_STEPS}")
    print(f"Learning Rate Scheduler: CosineAnnealingWarmRestarts")
    print(f"torch.compile: {'Enabled' if TORCH_COMPILE else 'Disabled'}")
    print(f"Track reward components: {'Enabled' if TRACK_REWARD_COMPONENTS else 'Disabled'}")
    print("=" * 70)
    print()
    print("ADVANCED OPTIMIZATIONS:")
    print("  - Double DQN: Reduces Q-value overestimation")
    print("  - Soft Target Updates: Smooth Polyak averaging (tau={})".format(TAU))
    print("  - N-Step Returns: {}-step TD learning for faster credit assignment".format(N_STEPS))
    print("  - One-Hot State Encoding: 16-channel representation for precise tile values")
    print("  - Batch Normalization: Training stability")
    print("  - Tuned Rewards: Log-merge + potential delta + tile progress")
    print("  - Learning Rate Scheduling: Cosine annealing with warm restarts")
    print("=" * 70)
    print()

# Estimated training time
if FAST_MODE:
    if USE_CUDA:
        if COLAB_SPEED_PROFILE:
            print(f"Estimated training time: 20-45 minutes (Colab GPU speed profile)")
        else:
            print(f"Estimated training time: 60-90 minutes (GPU)")
    else:
        print(f"Estimated training time: 4-6 hours (CPU)")
else:
    if USE_CUDA:
        if COLAB_SPEED_PROFILE:
            print(f"Estimated training time: 30-60 minutes (Colab GPU speed profile)")
        else:
            print(f"Estimated training time: 30-60 minutes (GPU)")
    else:
        print(f"Estimated training time: 3-5 hours (CPU)")
print()
