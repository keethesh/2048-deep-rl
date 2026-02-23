import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time

IN_COLAB = "COLAB_GPU" in os.environ

# Try to import config for optimizations, fall back to defaults
try:
    from config import (
        FAST_MODE, NUM_EPISODES, BATCH_SIZE, LEARNING_RATE,
        EPSILON_DECAY, EVAL_FREQ, EVAL_EPISODES,
        SAVE_FREQ, LOG_FREQ, MEMORY_SIZE, CONV_FILTERS, FC_SIZE,
        USE_MIXED_PRECISION, MAX_GRAD_NORM, DEVICE, print_config
    )
    USE_CONFIG = True
    # Try to import new config values with defaults
    try:
        from config import TRAIN_FREQ, TAU, N_STEPS
    except ImportError:
        TRAIN_FREQ = 4
        TAU = 0.005
        N_STEPS = 3
    try:
        from config import LR_SCHEDULER_T_0, LR_SCHEDULER_T_MULT, LR_SCHEDULER_ETA_MIN
    except ImportError:
        LR_SCHEDULER_T_0 = 1000
        LR_SCHEDULER_T_MULT = 2
        LR_SCHEDULER_ETA_MIN = 1e-6
    try:
        from config import PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES
    except ImportError:
        PER_ALPHA = 0.6
        PER_BETA_START = 0.4
        PER_BETA_FRAMES = 100000
    try:
        from config import (
            REWARD_INVALID_PENALTY,
            REWARD_MERGE_LOG_WEIGHT,
            REWARD_POTENTIAL_DELTA_WEIGHT,
            REWARD_TILE_PROGRESS_WEIGHT,
            REWARD_SURVIVAL_BONUS,
            REWARD_CLIP_MIN,
            REWARD_CLIP_MAX,
        )
    except ImportError:
        REWARD_INVALID_PENALTY = -1.5
        REWARD_MERGE_LOG_WEIGHT = 0.7
        REWARD_POTENTIAL_DELTA_WEIGHT = 2.5
        REWARD_TILE_PROGRESS_WEIGHT = 0.35
        REWARD_SURVIVAL_BONUS = 0.02
        REWARD_CLIP_MIN = -4.0
        REWARD_CLIP_MAX = 6.0
    try:
        from config import DETERMINISTIC
    except ImportError:
        DETERMINISTIC = False
    try:
        from config import SEED
    except ImportError:
        SEED = 42
    try:
        from config import TORCH_COMPILE
    except ImportError:
        TORCH_COMPILE = True
    try:
        from config import TRACK_REWARD_COMPONENTS
    except ImportError:
        TRACK_REWARD_COMPONENTS = False
    try:
        from config import EVAL_DEBUG_ON_FIRST
    except ImportError:
        EVAL_DEBUG_ON_FIRST = False
except ImportError:
    USE_CONFIG = False
    FAST_MODE = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_FREQ = 4
    TAU = 0.005
    N_STEPS = 3
    LR_SCHEDULER_T_0 = 1000
    LR_SCHEDULER_T_MULT = 2
    LR_SCHEDULER_ETA_MIN = 1e-6
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 100000
    REWARD_INVALID_PENALTY = -1.5
    REWARD_MERGE_LOG_WEIGHT = 0.7
    REWARD_POTENTIAL_DELTA_WEIGHT = 2.5
    REWARD_TILE_PROGRESS_WEIGHT = 0.35
    REWARD_SURVIVAL_BONUS = 0.02
    REWARD_CLIP_MIN = -4.0
    REWARD_CLIP_MAX = 6.0
    DETERMINISTIC = False
    SEED = 42
    TORCH_COMPILE = True
    TRACK_REWARD_COMPONENTS = False
    EVAL_DEBUG_ON_FIRST = False


def configure_randomness():
    """
    Configure reproducibility controls for Python, NumPy, and PyTorch.
    - Set TRAIN_SEED env var to force a deterministic seed.
    - Otherwise, seed only when DETERMINISTIC is enabled in config.
    """
    seed_override = os.environ.get("TRAIN_SEED")
    if seed_override is not None:
        try:
            seed = int(seed_override)
        except ValueError as exc:
            raise ValueError("TRAIN_SEED must be an integer.") from exc
    elif DETERMINISTIC:
        seed = int(SEED)
    else:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


RUN_SEED = configure_randomness()


# ============================================================================
# N-STEP RETURNS BUFFER
# ============================================================================

class NStepBuffer:
    """
    Buffer for accumulating N-step transitions.
    Computes multi-step returns: R = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^n*V(s_n)
    """
    def __init__(self, n_steps=3, gamma=0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque()
    
    def add(self, state, action, reward, next_state, done, next_valid_mask):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))
        
        # Once full, emit one overlapping n-step transition.
        if len(self.buffer) >= self.n_steps:
            return self._compute_n_step_return(self.n_steps)

        # If episode ended early, emit a short transition now.
        if done:
            return self._compute_n_step_return(len(self.buffer))
        return None
    
    def _compute_n_step_return(self, steps):
        """Compute n-step return from the first `steps` entries in the buffer."""
        if steps <= 0:
            return None

        transitions = list(self.buffer)[:steps]

        # Sum discounted rewards
        n_step_return = 0.0
        for i, (_, _, reward, _, _, _) in enumerate(transitions):
            n_step_return += (self.gamma ** i) * reward
        
        # Return:
        # (state_0, action_0, n_step_return, state_n, done_n, step_count, next_valid_mask_n)
        state_0, action_0, _, _, _, _ = transitions[0]
        _, _, _, state_n, done_n, next_valid_mask_n = transitions[-1]
        
        # Sliding window for overlapping n-step transitions
        self.buffer.popleft()
        
        return (state_0, action_0, n_step_return, state_n, done_n, steps, next_valid_mask_n)

    def flush(self):
        """
        Flush all remaining transitions at episode end.
        Produces short-tail transitions for proper terminal credit assignment.
        """
        transitions = []
        while self.buffer:
            steps = min(self.n_steps, len(self.buffer))
            transition = self._compute_n_step_return(steps)
            if transition is not None:
                transitions.append(transition)
        return transitions
    
    def reset(self):
        """Reset buffer (call when episode ends)."""
        self.buffer.clear()


# ============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# ============================================================================

class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    Stores priorities in a binary tree for O(log n) updates and sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples experiences based on their TD-error for more efficient learning.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling weight
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 0.01  # Small constant to avoid zero priority
        self.abs_error_upper = 1.0
        self.beta = beta_start
        self.max_priority = 1.0

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        if error is None:
            priority = self.max_priority
        else:
            priority = self._get_priority(error)
            self.max_priority = max(self.max_priority, priority)
        self.tree.add(priority, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        total_priority = max(self.tree.total(), 1e-8)
        segment = total_priority / batch_size
        priorities = []

        # Anneal beta from beta_start to 1
        self.beta = np.min([1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames])
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_priority
        is_weight = np.power(self.tree.n_entries * sampling_probabilities + 1e-8, -self.beta)
        max_w = is_weight.max()
        if max_w > 0:
            is_weight /= max_w
        else:
            is_weight = np.ones_like(is_weight)

        return batch, idxs, is_weight

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)
        self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


# ============================================================================
# STATE PREPROCESSING
# ============================================================================

def preprocess_state(board):
    """
    Convert raw board state to 16-channel one-hot encoding.
    Each channel represents a tile power: 0, 2, 4, 8, ..., 65536
    This preserves tile identity and allows the network to distinguish values precisely.
    """
    encoded = np.zeros((16, 4, 4), dtype=np.float32)

    board_int = board.astype(np.int64, copy=False)
    planes = np.zeros_like(board_int, dtype=np.int64)
    non_zero = board_int > 0
    if np.any(non_zero):
        # 2->1, 4->2, ..., 2048->11
        planes[non_zero] = np.clip(np.log2(board_int[non_zero]).astype(np.int64), 1, 15)

    row_idx, col_idx = np.indices(board_int.shape)
    encoded[planes, row_idx, col_idx] = 1.0
    return encoded


# ============================================================================
# REWARD SHAPING
# ============================================================================

class RewardShaper:
    """
    Reward shaping tuned for training stability.
    Uses three bounded components:
    - Merge score (log scaled)
    - Potential-based board quality delta
    - Small progress/survival nudges
    """
    POTENTIAL_EMPTY_WEIGHT = 1.2
    POTENTIAL_MONOTONICITY_WEIGHT = 0.45
    POTENTIAL_SMOOTHNESS_WEIGHT = 0.20
    POTENTIAL_CORNER_WEIGHT = 0.35
    POTENTIAL_MAX_TILE_WEIGHT = 0.55
    MAX_MONOTONICITY_PENALTY = 88.0
    MAX_SMOOTHNESS_PENALTY = 264.0

    @staticmethod
    def calculate_empty_tile_bonus(board):
        """Normalized empty-space bonus in [0, 1]."""
        empty_count = np.sum(board == 0)
        return empty_count / 16.0

    @staticmethod
    def _log2_board(board):
        """Compute log2 board representation with zeros preserved."""
        log_board = np.zeros_like(board, dtype=np.float32)
        non_zero = board > 0
        log_board[non_zero] = np.log2(board[non_zero])
        return log_board

    @staticmethod
    def _line_monotonic_penalty(line):
        """Lower is better; 0 means strictly monotonic after removing zeros."""
        non_zero = line[line > 0]
        if non_zero.size <= 1:
            return 0.0
        diffs = np.diff(non_zero)
        increase_penalty = np.sum(np.clip(diffs, 0.0, None))
        decrease_penalty = np.sum(np.clip(-diffs, 0.0, None))
        return float(min(increase_penalty, decrease_penalty))

    @staticmethod
    def calculate_monotonicity(board):
        """
        Normalized monotonicity score in [-1, 0].
        Closer to 0 means better directional consistency.
        """
        log_board = RewardShaper._log2_board(board)
        total_penalty = 0.0

        for row in log_board:
            total_penalty += RewardShaper._line_monotonic_penalty(row)
        for col in log_board.T:
            total_penalty += RewardShaper._line_monotonic_penalty(col)

        normalized = -min(total_penalty, RewardShaper.MAX_MONOTONICITY_PENALTY) / RewardShaper.MAX_MONOTONICITY_PENALTY
        return float(normalized)

    @staticmethod
    def calculate_smoothness(board):
        """
        Normalized smoothness score in [-1, 0].
        Closer to 0 means adjacent tiles are more compatible.
        """
        log_board = RewardShaper._log2_board(board)
        penalty = 0.0

        for i in range(4):
            for j in range(4):
                if board[i, j] == 0:
                    continue
                value = log_board[i, j]

                # Check right neighbor
                if j < 3 and board[i, j + 1] != 0:
                    penalty += abs(value - log_board[i, j + 1])
                # Check down neighbor
                if i < 3 and board[i + 1, j] != 0:
                    penalty += abs(value - log_board[i + 1, j])

        normalized = -min(penalty, RewardShaper.MAX_SMOOTHNESS_PENALTY) / RewardShaper.MAX_SMOOTHNESS_PENALTY
        return float(normalized)

    @staticmethod
    def calculate_corner_bonus(board):
        """Returns 1.0 if the max tile is in a corner, else 0.0."""
        max_val = np.max(board)
        if max_val == 0:
            return 0.0
        corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
        if max_val in corners:
            return 1.0
        return 0.0

    @staticmethod
    def calculate_max_tile_progress(prev_board, board):
        """
        Returns positive log-scale progress when max tile improves.
        A single tile doubling yields +1.0.
        """
        prev_max = int(np.max(prev_board))
        current_max = int(np.max(board))
        if current_max <= prev_max:
            return 0.0

        baseline = max(prev_max, 2)
        return float(np.log2(current_max / baseline))

    @staticmethod
    def calculate_board_potential(board, return_components=False):
        """
        Board quality potential for potential-based shaping.
        Higher is better.
        """
        empty_ratio = RewardShaper.calculate_empty_tile_bonus(board)
        monotonicity = RewardShaper.calculate_monotonicity(board)
        smoothness = RewardShaper.calculate_smoothness(board)
        corner_bonus = RewardShaper.calculate_corner_bonus(board)

        max_tile = int(np.max(board))
        max_tile_norm = 0.0 if max_tile == 0 else min(np.log2(max_tile) / 16.0, 1.0)

        potential = (
            RewardShaper.POTENTIAL_EMPTY_WEIGHT * empty_ratio
            + RewardShaper.POTENTIAL_MONOTONICITY_WEIGHT * monotonicity
            + RewardShaper.POTENTIAL_SMOOTHNESS_WEIGHT * smoothness
            + RewardShaper.POTENTIAL_CORNER_WEIGHT * corner_bonus
            + RewardShaper.POTENTIAL_MAX_TILE_WEIGHT * max_tile_norm
        )

        if not return_components:
            return potential

        components = {
            "empty_ratio": empty_ratio,
            "monotonicity": monotonicity,
            "smoothness": smoothness,
            "corner_bonus": corner_bonus,
            "max_tile_norm": max_tile_norm,
            "potential": potential,
        }
        return potential, components

    @staticmethod
    def shape_reward(board, base_reward, board_changed, prev_board=None, return_components=False):
        """
        Tuned reward shaping:
        reward = merge_log + potential_delta + tile_progress + survival_bonus
        Final reward is clipped to avoid extreme Q-targets.
        """
        if prev_board is None:
            prev_board = board

        if not board_changed:
            reward = float(REWARD_INVALID_PENALTY)
            if return_components:
                components = {
                    "merge_reward": 0.0,
                    "potential_before": 0.0,
                    "potential_after": 0.0,
                    "potential_delta": 0.0,
                    "tile_progress": 0.0,
                    "survival_bonus": 0.0,
                    "raw_reward": reward,
                    "clipped_reward": reward,
                }
                return reward, components
            return reward

        merge_reward = REWARD_MERGE_LOG_WEIGHT * np.log1p(base_reward)
        if return_components:
            potential_before, before_components = RewardShaper.calculate_board_potential(
                prev_board,
                return_components=True,
            )
            potential_after, after_components = RewardShaper.calculate_board_potential(
                board,
                return_components=True,
            )
        else:
            potential_before = RewardShaper.calculate_board_potential(prev_board)
            potential_after = RewardShaper.calculate_board_potential(board)

        potential_delta = potential_after - potential_before
        tile_progress = RewardShaper.calculate_max_tile_progress(prev_board, board)
        survival_bonus = REWARD_SURVIVAL_BONUS

        raw_reward = (
            merge_reward
            + REWARD_POTENTIAL_DELTA_WEIGHT * potential_delta
            + REWARD_TILE_PROGRESS_WEIGHT * tile_progress
            + survival_bonus
        )

        reward = float(np.clip(raw_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

        if return_components:
            components = {
                "merge_reward": float(merge_reward),
                "potential_before": float(potential_before),
                "potential_after": float(potential_after),
                "potential_delta": float(potential_delta),
                "tile_progress": float(tile_progress),
                "survival_bonus": float(survival_bonus),
                "raw_reward": float(raw_reward),
                "clipped_reward": reward,
                "board_features_before": before_components,
                "board_features_after": after_components,
            }
            return reward, components
        return reward


# ============================================================================
# DUELING CNN-DQN ARCHITECTURE
# ============================================================================

class DuelingCNNDQN(nn.Module):
    """
    Dueling DQN with Convolutional layers for spatial feature extraction.

    Architecture:
    - Conv layers to capture spatial patterns (corners, edges, monotonicity)
    - Batch normalization for training stability
    - Dueling streams: Value function V(s) and Advantage function A(s,a)
    - Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    This separates "how good is this state" from "how much better is each action".
    """
    def __init__(self, input_channels=16, output_dim=4, conv_filters=None, fc_size=None):
        super(DuelingCNNDQN, self).__init__()

        # Use config values if available, otherwise defaults
        if conv_filters is None:
            conv_filters = CONV_FILTERS if USE_CONFIG else 256
        if fc_size is None:
            fc_size = FC_SIZE if USE_CONFIG else 512

        # Convolutional feature extractor with batch normalization
        # Input: (batch, 16, 4, 4) for one-hot encoded states
        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(conv_filters)
        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(conv_filters)

        # Calculate conv output size: (batch, conv_filters, 1, 1) after 3x 2x2 convs
        conv_output_size = conv_filters * 1 * 1

        # Dueling streams
        # Value stream: V(s)
        self.value_fc1 = nn.Linear(conv_output_size, fc_size)
        self.value_fc2 = nn.Linear(fc_size, 1)

        # Advantage stream: A(s, a)
        self.advantage_fc1 = nn.Linear(conv_output_size, fc_size)
        self.advantage_fc2 = nn.Linear(fc_size, output_dim)

    def forward(self, x):
        # Handle different input shapes
        if len(x.shape) == 2:
            # Flattened input: (batch, 64) -> (batch, 16, 4, 4)
            x = x.view(-1, 16, 4, 4)
        elif len(x.shape) == 4:
            # Already in shape (batch, channels, height, width)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Convolutional feature extraction with batch norm
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Value stream
        value = torch.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # Advantage stream
        advantage = torch.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


# ============================================================================
# GAME ENVIRONMENT
# ============================================================================

class Game2048:
    def __init__(self, seed=None, rng=None):
        self.board = np.zeros((4, 4), dtype=int)
        self.rng = rng if rng is not None else (random.Random(seed) if seed is not None else random)
        self._valid_actions_cache = None
        self._valid_action_mask_cache = None
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        self._invalidate_valid_actions_cache()
        return preprocess_state(self.board)  # Return (16, 4, 4) shape

    def _invalidate_valid_actions_cache(self):
        self._valid_actions_cache = None
        self._valid_action_mask_cache = None

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = self.rng.choice(empty_cells)
            self.board[row, col] = 2 if self.rng.random() < 0.9 else 4
            self._invalidate_valid_actions_cache()

    def slide_and_merge_row_left(self, row):
        new_row = row[row != 0]
        if len(new_row) == 0:
            return row, 0

        score = 0
        merged_row = []
        skip = False

        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
                merged_row.append(new_row[i] * 2)
                score += new_row[i] * 2
                skip = True
            else:
                merged_row.append(new_row[i])

        merged_row.extend([0] * (len(row) - len(merged_row)))
        return np.array(merged_row), score

    def move_board(self, direction):
        original_board = self.board.copy()
        total_score = 0

        if direction == 0:  # Up
            self.board = np.rot90(self.board, 1)
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
            self.board = np.rot90(self.board, -1)
        elif direction == 1:  # Down
            self.board = np.rot90(self.board, -1)
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
            self.board = np.rot90(self.board, 1)
        elif direction == 2:  # Left
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
        elif direction == 3:  # Right
            self.board = np.rot90(self.board, 2)
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
            self.board = np.rot90(self.board, 2)
        else:
            raise ValueError(f"Invalid action: {direction}")

        board_changed = not np.array_equal(original_board, self.board)
        if board_changed:
            self._invalidate_valid_actions_cache()
        return total_score, board_changed

    def get_valid_actions(self):
        """
        Return list of actions that change the board.
        Optimized to check without actually modifying the board.
        """
        if self._valid_actions_cache is None:
            mask = self._compute_valid_action_mask()
            self._valid_action_mask_cache = mask
            self._valid_actions_cache = np.flatnonzero(mask).tolist()
        return list(self._valid_actions_cache)

    def get_valid_action_mask(self):
        """Return boolean mask of valid actions in [Up, Down, Left, Right] order."""
        if self._valid_action_mask_cache is None:
            self.get_valid_actions()
        return self._valid_action_mask_cache.copy()

    def _compute_valid_action_mask(self):
        mask = np.zeros(4, dtype=bool)
        for action in range(4):
            mask[action] = self._would_move_change_board(action)
        return mask

    def _would_move_change_board(self, direction):
        """
        Fast check if a move would change the board without actually doing it.
        """
        # Create a temporary rotated view (matching move_board rotation)
        if direction == 0:  # Up
            temp_board = np.rot90(self.board, 1)
        elif direction == 1:  # Down
            temp_board = np.rot90(self.board, -1)
        elif direction == 2:  # Left
            temp_board = self.board
        elif direction == 3:  # Right
            temp_board = np.rot90(self.board, 2)
        else:
            return False

        # Check if any row would change when slid left
        for row in temp_board:
            non_zero = row[row != 0]
            if len(non_zero) == 0:
                continue

            # Check if any adjacent tiles can merge
            for i in range(len(non_zero) - 1):
                if non_zero[i] == non_zero[i + 1]:
                    return True  # Can merge

            # Check if tiles are already in leftmost positions
            # Compare non-zero elements with leftmost positions of row
            for i in range(len(non_zero)):
                if row[i] != non_zero[i]:
                    return True  # Tiles would move to different positions

        return False

    def step(self, action, return_info=False, include_reward_components=False):
        board_before_move = self.board.copy()
        score, board_changed = self.move_board(action)

        # Advanced reward shaping
        if return_info and include_reward_components:
            reward, reward_components = RewardShaper.shape_reward(
                self.board,
                score,
                board_changed,
                prev_board=board_before_move,
                return_components=True,
            )
        else:
            reward = RewardShaper.shape_reward(
                self.board,
                score,
                board_changed,
                prev_board=board_before_move,
                return_components=False,
            )
            reward_components = None

        if board_changed:
            self.add_random_tile()

        done = self.is_game_over()
        if done:
            valid_actions = []
            valid_action_mask = np.zeros(4, dtype=bool)
            self._valid_actions_cache = []
            self._valid_action_mask_cache = valid_action_mask
        else:
            valid_actions = self.get_valid_actions()
            valid_action_mask = self.get_valid_action_mask()

        next_state = preprocess_state(self.board)  # Return (16, 4, 4) shape

        if return_info:
            info = {
                "base_score": score,
                "board_changed": board_changed,
                "valid_actions": valid_actions,
                "valid_action_mask": valid_action_mask,
                "max_tile": int(np.max(self.board)),
                "reward_components": reward_components,
            }
            return next_state, reward, done, info

        return next_state, reward, done

    def is_game_over(self):
        if np.any(self.board == 0):
            return False

        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return False
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return False

        return True


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Use config values if available, otherwise use defaults
if USE_CONFIG:
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    memory_size = MEMORY_SIZE
    epsilon_decay = EPSILON_DECAY
    device = torch.device(DEVICE)
    train_freq = TRAIN_FREQ
    tau = TAU
    n_steps = N_STEPS
    per_alpha = PER_ALPHA
    per_beta_start = PER_BETA_START
    per_beta_frames = PER_BETA_FRAMES
else:
    learning_rate = 0.0001
    batch_size = 64
    memory_size = 50000
    epsilon_decay = 0.995
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_freq = 4
    tau = 0.005
    n_steps = 3
    per_alpha = 0.6
    per_beta_start = 0.4
    per_beta_frames = 100000

# Common hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01

print(f"Using device: {device}")
if USE_CONFIG:
    print(f"Config loaded: {'FAST' if FAST_MODE else 'FULL'} mode")
    print_config()
print(
    "Reward shaping: "
    f"invalid={REWARD_INVALID_PENALTY}, merge_w={REWARD_MERGE_LOG_WEIGHT}, "
    f"delta_w={REWARD_POTENTIAL_DELTA_WEIGHT}, tile_w={REWARD_TILE_PROGRESS_WEIGHT}, "
    f"clip=[{REWARD_CLIP_MIN}, {REWARD_CLIP_MAX}]"
)
if RUN_SEED is not None:
    print(f"Reproducibility seed: {RUN_SEED}")

# CUDA backend speedups with no algorithmic behavior change.
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not DETERMINISTIC:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print("CUDA backend optimizations: TF32 + cuDNN benchmark enabled")

# ============================================================================
# INITIALIZE
# ============================================================================

env = Game2048(seed=RUN_SEED)
state_dim = 16
action_dim = 4

# Initialize networks with 16-channel input for one-hot encoding
q_network = DuelingCNNDQN(input_channels=16, output_dim=action_dim).to(device)
target_network = DuelingCNNDQN(input_channels=16, output_dim=action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

enable_torch_compile = bool(TORCH_COMPILE and hasattr(torch, "compile") and device.type == "cuda")
if enable_torch_compile:
    try:
        q_network = torch.compile(q_network, mode="default")
        target_network = torch.compile(target_network, mode="default")
        print("torch.compile: ENABLED")
    except Exception as exc:
        enable_torch_compile = False
        print(f"torch.compile: DISABLED ({exc})")
else:
    print("torch.compile: DISABLED")

def unwrap_model(model):
    """Return underlying nn.Module when wrapped by torch.compile."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


q_network_base = unwrap_model(q_network)
target_network_base = unwrap_model(target_network)

optimizer = optim.Adam(q_network_base.parameters(), lr=learning_rate)

# Learning rate scheduler with cosine annealing warm restarts
if USE_CONFIG:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_SCHEDULER_T_0,
        T_mult=LR_SCHEDULER_T_MULT,
        eta_min=LR_SCHEDULER_ETA_MIN
    )
else:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_SCHEDULER_T_0,
        T_mult=LR_SCHEDULER_T_MULT,
        eta_min=LR_SCHEDULER_ETA_MIN
    )

memory = PrioritizedReplayBuffer(
    memory_size,
    alpha=per_alpha,
    beta_start=per_beta_start,
    beta_frames=per_beta_frames,
)
n_step_buffer = NStepBuffer(n_steps=n_steps, gamma=gamma)

# Mixed precision training (GPU only)
use_amp = USE_CONFIG and USE_MIXED_PRECISION and device.type == "cuda"
if use_amp:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    print("Mixed precision training: ENABLED")
else:
    scaler = None
    if USE_CONFIG:
        print("Mixed precision training: DISABLED (CPU or disabled in config)")

# TensorBoard
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
mode_suffix = f"_{'fast' if USE_CONFIG and FAST_MODE else 'full'}"
log_dir = f"runs/2048_advanced{mode_suffix}_{timestamp}"
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs: {log_dir}")
if RUN_SEED is not None:
    writer.add_text("Run/Seed", str(RUN_SEED), 0)
writer.add_text("Run/Deterministic", str(DETERMINISTIC), 0)
writer.add_text("Run/PER", f"alpha={per_alpha}, beta_start={per_beta_start}, beta_frames={per_beta_frames}", 0)
writer.add_text("Run/Colab", str(IN_COLAB), 0)
writer.add_text("Run/TorchCompile", str(enable_torch_compile), 0)
writer.add_text("Run/TrackRewardComponents", str(TRACK_REWARD_COMPONENTS), 0)
writer.add_text(
    "Run/RewardShaping",
    (
        f"invalid={REWARD_INVALID_PENALTY}, merge_w={REWARD_MERGE_LOG_WEIGHT}, "
        f"delta_w={REWARD_POTENTIAL_DELTA_WEIGHT}, tile_w={REWARD_TILE_PROGRESS_WEIGHT}, "
        f"survival={REWARD_SURVIVAL_BONUS}, clip=[{REWARD_CLIP_MIN}, {REWARD_CLIP_MAX}]"
    ),
    0,
)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def choose_action(state, epsilon, valid_actions=None):
    """
    Epsilon-greedy action selection with optional action masking.
    Handles both flattened and multi-channel state representations.
    """
    if random.random() < epsilon:
        if valid_actions is not None and len(valid_actions) > 0:
            return random.choice(valid_actions)
        return random.randint(0, action_dim - 1)

    # Convert state to tensor - handle both (16, 4, 4) and flattened shapes
    if isinstance(state, np.ndarray):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        elif state_tensor.dim() == 1:
            state_tensor = state_tensor.view(1, 16, 4, 4)
        elif state_tensor.dim() != 4:
            raise ValueError(f"Unexpected numpy state shape: {state.shape}")
    else:
        state_tensor = state.to(device)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
    
    # Set network to eval mode for inference (BatchNorm compatibility)
    was_training = q_network.training
    if was_training:
        q_network.eval()
    with torch.inference_mode():
        if use_amp:
            with autocast():
                q_values = q_network(state_tensor)
        else:
            q_values = q_network(state_tensor)
    if was_training:
        q_network.train()

    # If valid actions provided, mask invalid ones
    if valid_actions is not None and len(valid_actions) > 0:
        valid_idx = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        valid_mask = torch.zeros(action_dim, dtype=torch.bool, device=device)
        valid_mask[valid_idx] = True
        masked_q = q_values.squeeze(0).masked_fill(~valid_mask, -1e9)
        return int(masked_q.argmax().item())

    return int(q_values.argmax(dim=1).item())


def train_step():
    """
    Perform one training step using prioritized experience replay with Double DQN.
    Supports mixed precision training for GPU acceleration.
    """
    if len(memory) < batch_size:
        return None

    batch, idxs, is_weights = memory.sample(batch_size)
    states, actions, rewards, next_states, dones, step_counts, next_valid_masks = zip(*batch)

    # Convert batch to tensors
    states = torch.from_numpy(np.stack(states).astype(np.float32, copy=False)).to(device)
    next_states = torch.from_numpy(np.stack(next_states).astype(np.float32, copy=False)).to(device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    step_counts = torch.as_tensor(step_counts, dtype=torch.float32, device=device)
    next_valid_masks = torch.as_tensor(np.asarray(next_valid_masks, dtype=bool), dtype=torch.bool, device=device)
    is_weights = torch.as_tensor(is_weights, dtype=torch.float32, device=device)

    discount_factors = torch.pow(torch.full_like(step_counts, gamma), step_counts)
    has_valid_next_action = next_valid_masks.any(dim=1).float()

    # Use mixed precision if enabled
    if use_amp:
        with autocast():
            # Current Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN with next-state valid-action masking
            with torch.no_grad():
                next_policy_q_values = q_network(next_states)
                masked_next_policy_q_values = next_policy_q_values.masked_fill(~next_valid_masks, -1e9)
                next_actions = masked_next_policy_q_values.argmax(dim=1, keepdim=True)
                next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
                next_q_values = next_q_values * has_valid_next_action
                target_q_values = rewards + discount_factors * next_q_values * (1 - dones)

            # Weighted loss
            loss = (is_weights * (current_q_values - target_q_values).pow(2)).mean()

        # TD errors for priority updates
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()

        # Scaled backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        max_grad = MAX_GRAD_NORM if USE_CONFIG else 10.0
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_grad)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard precision
        # Current Q values
        current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN with next-state valid-action masking
        with torch.no_grad():
            next_policy_q_values = q_network(next_states)
            masked_next_policy_q_values = next_policy_q_values.masked_fill(~next_valid_masks, -1e9)
            next_actions = masked_next_policy_q_values.argmax(dim=1, keepdim=True)
            next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
            next_q_values = next_q_values * has_valid_next_action
            target_q_values = rewards + discount_factors * next_q_values * (1 - dones)

        # TD errors for priority updates
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()

        # Weighted loss
        loss = (is_weights * (current_q_values - target_q_values).pow(2)).mean()

        optimizer.zero_grad()
        loss.backward()
        max_grad = MAX_GRAD_NORM if USE_CONFIG else 10.0
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_grad)
        optimizer.step()

    # Soft target network update (Polyak averaging)
    for target_param, param in zip(target_network_base.parameters(), q_network_base.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # Update priorities in replay buffer
    for idx, error in zip(idxs, td_errors):
        memory.update(idx, error)

    return loss.item()


def evaluate(num_episodes=10, debug=False):
    """
    Evaluate the agent without exploration (epsilon=0).
    Returns average reward and average max tile achieved.
    """
    # Set network to eval mode for evaluation
    q_network.eval()
    
    total_rewards = []
    max_tiles = []

    # Track invalid move statistics
    invalid_move_counts = []

    for ep_num in range(num_episodes):
        state = env.reset()
        valid_actions = env.get_valid_actions()
        done = False
        episode_reward = 0
        moves = 0
        invalid_moves = 0
        max_moves = 10000  # Safety limit to prevent infinite loops

        if debug and ep_num == 0:
            print(f"\n{'='*80}")
            print(f"DEBUG EVALUATION - Episode {ep_num}")
            print(f"{'='*80}")
            print(f"Initial board:\n{env.board}")
            print(f"Initial preprocessed state shape: {state.shape}")

        while not done and moves < max_moves:
            # If no valid actions, game is stuck - end episode
            if not valid_actions:
                if debug and ep_num == 0:
                    print(f"\nMove {moves}: No valid actions - ending episode")
                done = True
                break

            board_before = env.board.copy() if (debug and ep_num == 0 and moves < 20) else None

            current_valid_actions = valid_actions
            action = choose_action(state, epsilon=0, valid_actions=current_valid_actions)
            next_state, reward, done, info = env.step(
                action,
                return_info=True,
                include_reward_components=debug,
            )
            board_changed = info["board_changed"]
            valid_actions = info["valid_actions"]

            # Detect invalid moves directly from environment transition info
            if not board_changed:
                invalid_moves += 1

            if debug and ep_num == 0 and moves < 20:  # Log first 20 moves
                board_after = env.board.copy()
                action_names = ['Up', 'Down', 'Left', 'Right']
                print(f"\n--- Move {moves} ---")
                print(f"Valid actions: {[action_names[a] for a in current_valid_actions]}")
                print(f"Chosen action: {action_names[action]} (index: {action})")
                print(f"Board BEFORE:\n{board_before}")
                print(f"Board AFTER:\n{board_after}")
                print(f"Board changed: {board_changed}")
                print(f"Reward: {reward}")
                reward_components = info.get("reward_components")
                if reward_components is not None:
                    print(
                        "Reward terms: "
                        f"merge={reward_components['merge_reward']:.3f}, "
                        f"delta={reward_components['potential_delta']:.3f}, "
                        f"tile={reward_components['tile_progress']:.3f}, "
                        f"raw={reward_components['raw_reward']:.3f}, "
                        f"clip={reward_components['clipped_reward']:.3f}"
                    )
                print(f"Max tile: {info['max_tile']}")

            state = next_state
            episode_reward += reward
            moves += 1

        total_rewards.append(episode_reward)
        max_tiles.append(np.max(env.board))
        invalid_move_counts.append(invalid_moves)

        if debug and ep_num == 0:
            print(f"\n{'='*80}")
            print(f"Episode {ep_num} Summary:")
            print(f"Total moves: {moves}")
            print(f"Invalid moves: {invalid_moves}")
            print(f"Total reward: {episode_reward}")
            print(f"Max tile: {np.max(env.board)}")
            print(f"{'='*80}\n")

    if debug:
        print(f"\nAverage invalid moves per episode: {np.mean(invalid_move_counts):.1f}")
        print(f"Max invalid moves in an episode: {np.max(invalid_move_counts)}")
        print(f"Min invalid moves in an episode: {np.min(invalid_move_counts)}")

    # Set network back to train mode
    q_network.train()
    
    return np.mean(total_rewards), np.mean(max_tiles), np.max(max_tiles)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    global epsilon

    # Use config values if available
    if USE_CONFIG:
        num_episodes = NUM_EPISODES
        eval_freq = EVAL_FREQ
        eval_episodes = EVAL_EPISODES
        save_freq = SAVE_FREQ
        log_freq = LOG_FREQ
    else:
        num_episodes = 5000
        eval_freq = 50
        eval_episodes = 10
        save_freq = 100
        log_freq = 1

    best_avg_tile = 0
    os.makedirs("models", exist_ok=True)

    # Training start info
    if not USE_CONFIG or not FAST_MODE:
        print(f"\nStarting Advanced 2048 AI Training")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Network: Dueling CNN-DQN (16-channel one-hot)")
        print(f"Memory: Prioritized Experience Replay ({memory_size})")
        print(f"PER alpha/beta_start: {per_alpha}/{per_beta_start}")
        print(f"State: 16-channel one-hot encoding")
        print(f"Rewards: Tuned (log-merge + potential delta + tile progress)")
        print(f"N-step returns: {n_steps}")
        print(f"Training frequency: Every {train_freq} steps")
        print(f"Soft target updates: tau={tau}")
        print(f"torch.compile: {'ON' if enable_torch_compile else 'OFF'}")
        print(f"Track reward components: {'ON' if TRACK_REWARD_COMPONENTS else 'OFF'}")
        if RUN_SEED is not None:
            print(f"Seed: {RUN_SEED} (deterministic={DETERMINISTIC})")
        print(f"{'='*60}\n")

    # Track training start time
    training_start_time = time.time()
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        valid_actions = env.get_valid_actions()
        n_step_buffer.reset()  # Reset n-step buffer at episode start
        done = False
        total_reward = 0
        moves = 0
        losses = []
        reward_component_totals = None
        if TRACK_REWARD_COMPONENTS:
            reward_component_totals = {
                "merge_reward": 0.0,
                "potential_delta": 0.0,
                "tile_progress": 0.0,
                "raw_reward": 0.0,
                "clipped_reward": 0.0,
            }
        reward_component_steps = 0
        max_moves = 10000  # Safety limit to prevent infinite loops

        while not done and moves < max_moves:
            # If no valid actions, game should be over
            if not valid_actions:
                done = True
                break

            action = choose_action(state, epsilon, valid_actions)
            next_state, reward, done, info = env.step(
                action,
                return_info=True,
                include_reward_components=TRACK_REWARD_COMPONENTS,
            )

            # Add to n-step buffer
            n_step_transition = n_step_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
                info["valid_action_mask"],
            )
            
            # If n-step buffer is full, add transition to replay buffer
            if n_step_transition is not None:
                memory.add(None, n_step_transition)

            state = next_state
            valid_actions = info["valid_actions"]
            total_reward += reward
            moves += 1
            total_steps += 1

            reward_components = info.get("reward_components")
            if reward_component_totals is not None and reward_components is not None:
                for key in reward_component_totals:
                    reward_component_totals[key] += float(reward_components[key])
                reward_component_steps += 1

            # Train every train_freq steps
            if total_steps % train_freq == 0:
                loss = train_step()
                if loss is not None:
                    losses.append(loss)
                    # Step learning rate scheduler
                    scheduler.step()

        # Handle remaining transitions in n-step buffer at episode end
        for transition in n_step_buffer.flush():
            memory.add(None, transition)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Logging
        max_tile = np.max(env.board)
        avg_loss = np.mean(losses) if losses else 0
        avg_reward_components = {}
        if reward_component_totals is not None and reward_component_steps > 0:
            avg_reward_components = {
                key: value / reward_component_steps
                for key, value in reward_component_totals.items()
            }

        # TensorBoard logging
        writer.add_scalar("Training/Reward", total_reward, episode)
        writer.add_scalar("Training/MaxTile", max_tile, episode)
        writer.add_scalar("Training/Moves", moves, episode)
        writer.add_scalar("Training/Epsilon", epsilon, episode)
        writer.add_scalar("Training/Loss", avg_loss, episode)
        writer.add_scalar("Training/LearningRate", scheduler.get_last_lr()[0], episode)
        if avg_reward_components:
            writer.add_scalar("Reward/AvgMerge", avg_reward_components["merge_reward"], episode)
            writer.add_scalar("Reward/AvgPotentialDelta", avg_reward_components["potential_delta"], episode)
            writer.add_scalar("Reward/AvgTileProgress", avg_reward_components["tile_progress"], episode)
            writer.add_scalar("Reward/AvgRaw", avg_reward_components["raw_reward"], episode)
            writer.add_scalar("Reward/AvgClipped", avg_reward_components["clipped_reward"], episode)

        # Console logging (less frequent in fast mode)
        if episode % log_freq == 0 or episode == num_episodes - 1:
            # Calculate time estimates
            elapsed = time.time() - training_start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            remaining_eps = num_episodes - episode - 1
            eta_seconds = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
            eta_mins = eta_seconds / 60

            print(f"Ep {episode:4d}/{num_episodes} | Reward: {total_reward:7.1f} | MaxTile: {max_tile:4.0f} | "
                  f"Moves: {moves:3d} | Eps: {epsilon:.3f} | Loss: {avg_loss:.4f} | "
                  f"Speed: {eps_per_sec:.1f} ep/s | ETA: {eta_mins:.1f}m")

        # Evaluation
        if episode % eval_freq == 0 and episode > 0:
            debug_mode = EVAL_DEBUG_ON_FIRST and (episode == eval_freq)
            eval_reward, eval_avg_tile, eval_max_tile = evaluate(num_episodes=eval_episodes, debug=debug_mode)
            writer.add_scalar("Evaluation/AvgReward", eval_reward, episode)
            writer.add_scalar("Evaluation/AvgMaxTile", eval_avg_tile, episode)
            writer.add_scalar("Evaluation/MaxTile", eval_max_tile, episode)

            print(f"\n{'='*60}")
            print(f"EVALUATION (Episode {episode})")
            print(f"Avg Reward: {eval_reward:.1f} | Avg Max Tile: {eval_avg_tile:.1f} | Best: {eval_max_tile:.0f}")
            print(f"{'='*60}\n")

            # Save best model
            if eval_avg_tile > best_avg_tile:
                best_avg_tile = eval_avg_tile
                torch.save(q_network_base.state_dict(), f"models/best_model_{int(eval_avg_tile)}.pth")
                print(f"🎉 New best model saved! Avg tile: {eval_avg_tile:.1f}\n")

        # Periodic save
        if episode % save_freq == 0 and episode > 0:
            torch.save(q_network_base.state_dict(), f"models/checkpoint_ep{episode}.pth")

    # Final save
    torch.save(q_network_base.state_dict(), "models/final_model.pth")
    writer.close()

    # Training completion summary
    total_time = time.time() - training_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Average speed: {num_episodes/total_time:.2f} episodes/second")
    print(f"Models saved in ./models/")
    print(f"TensorBoard logs: {log_dir}")
    print(f"View with: tensorboard --logdir=runs")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
