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

# Try to import config for optimizations, fall back to defaults
try:
    from config import (
        FAST_MODE, NUM_EPISODES, BATCH_SIZE, LEARNING_RATE,
        EPSILON_DECAY, TARGET_UPDATE_FREQ, EVAL_FREQ, EVAL_EPISODES,
        SAVE_FREQ, LOG_FREQ, MEMORY_SIZE, CONV_FILTERS, FC_SIZE,
        USE_MIXED_PRECISION, MAX_GRAD_NORM, DEVICE, print_config
    )
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    FAST_MODE = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        priority = self._get_priority(error)
        self.tree.add(priority, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
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
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ============================================================================
# STATE PREPROCESSING
# ============================================================================

def preprocess_state(board):
    """
    Convert raw board state to log2-normalized representation.
    Maps: 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, ..., 2048 -> 11
    Then normalize to [0, 1] range for stable neural network training.
    """
    board = board.copy()
    # Apply log2 transformation (add 1 to handle zeros)
    mask = board > 0
    board[mask] = np.log2(board[mask])
    # Normalize to [0, 1] (max value is log2(131072) = 17 for safety)
    board = board / 17.0
    return board


# ============================================================================
# REWARD SHAPING
# ============================================================================

class RewardShaper:
    """
    Advanced reward shaping for 2048.
    Rewards behaviors that lead to better game states:
    - Empty tiles: More freedom to move
    - Monotonicity: High tiles organized in rows/columns
    - Smoothness: Adjacent tiles have similar values
    """
    @staticmethod
    def calculate_empty_tile_bonus(board):
        """Reward having more empty tiles (more freedom)."""
        empty_count = np.sum(board == 0)
        return empty_count * 10

    @staticmethod
    def calculate_monotonicity(board):
        """
        Reward monotonic rows/columns (descending or ascending).
        Encourages keeping high tiles organized.
        """
        mono_score = 0
        # Check rows
        for row in board:
            mono_score += RewardShaper._monotonicity_1d(row)
        # Check columns
        for col in board.T:
            mono_score += RewardShaper._monotonicity_1d(col)
        return mono_score

    @staticmethod
    def _monotonicity_1d(line):
        """Calculate monotonicity score for a single line."""
        non_zero = line[line > 0]
        if len(non_zero) <= 1:
            return 0

        # Check if line is monotonic increasing or decreasing
        increasing = np.all(non_zero[:-1] <= non_zero[1:])
        decreasing = np.all(non_zero[:-1] >= non_zero[1:])

        if increasing or decreasing:
            return np.sum(non_zero)
        return 0

    @staticmethod
    def calculate_smoothness(board):
        """
        Penalty for large differences between adjacent tiles.
        Encourages grouping similar values together.
        """
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] != 0:
                    value = np.log2(board[i][j]) if board[i][j] > 0 else 0
                    # Check right neighbor
                    if j < 3 and board[i][j+1] != 0:
                        target = np.log2(board[i][j+1])
                        smoothness -= abs(value - target)
                    # Check down neighbor
                    if i < 3 and board[i+1][j] != 0:
                        target = np.log2(board[i+1][j])
                        smoothness -= abs(value - target)
        return smoothness

    @staticmethod
    def calculate_corner_bonus(board):
        """Reward keeping the max tile in a corner."""
        max_val = np.max(board)
        corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
        if max_val in corners:
            return max_val
        return 0

    @staticmethod
    def shape_reward(board, base_reward, board_changed):
        """
        Combine all reward components.
        SIMPLIFIED version with less aggressive penalties.
        """
        if not board_changed:
            return -5  # Lighter penalty for invalid moves (was -50)

        # Start with merge score (already positive and meaningful)
        reward = base_reward

        # Empty tiles bonus (encourages keeping board open)
        empty_count = np.sum(board == 0)
        reward += empty_count * 2  # Simple linear bonus (was 10 * 0.5 = 5)

        # Max tile bonus (LOG-SCALED to match exponential tile growth)
        # This prevents huge reward spikes that destabilize Q-learning
        # Old: 20, 100, 500, 2000, 10000 (2500x variance)
        # New: log2-scaled with small multiplier (1.5x variance)
        max_tile = np.max(board)
        if max_tile > 0:
            # log2(2) = 1, log2(4) = 2, ..., log2(2048) = 11
            # Multiply by 2 to give meaningful reward signal
            tile_bonus = np.log2(max_tile) * 2
            reward += tile_bonus
            # Examples:
            # 128 -> log2(128) * 2 = 7 * 2 = 14
            # 256 -> log2(256) * 2 = 8 * 2 = 16
            # 512 -> log2(512) * 2 = 9 * 2 = 18
            # 1024 -> log2(1024) * 2 = 10 * 2 = 20
            # 2048 -> log2(2048) * 2 = 11 * 2 = 22

        # Small bonus for monotonicity (optional, scaled down)
        monotonicity = RewardShaper.calculate_monotonicity(board)
        reward += monotonicity * 0.001  # Very small contribution (was 0.01)

        # Corner bonus - encourage keeping max in corner
        corner_bonus = RewardShaper.calculate_corner_bonus(board)
        reward += corner_bonus * 0.05  # Small bonus (was 0.1)

        # REMOVED: Smoothness penalty (was causing huge negatives)
        # The merge score already encourages grouping similar tiles

        return reward


# ============================================================================
# DUELING CNN-DQN ARCHITECTURE
# ============================================================================

class DuelingCNNDQN(nn.Module):
    """
    Dueling DQN with Convolutional layers for spatial feature extraction.

    Architecture:
    - Conv layers to capture spatial patterns (corners, edges, monotonicity)
    - Dueling streams: Value function V(s) and Advantage function A(s,a)
    - Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    This separates "how good is this state" from "how much better is each action".
    """
    def __init__(self, input_channels=1, output_dim=4, conv_filters=None, fc_size=None):
        super(DuelingCNNDQN, self).__init__()

        # Use config values if available, otherwise defaults
        if conv_filters is None:
            conv_filters = CONV_FILTERS if USE_CONFIG else 128
        if fc_size is None:
            fc_size = FC_SIZE if USE_CONFIG else 256

        # Convolutional feature extractor
        # Input: (batch, 1, 4, 4)
        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1, padding=0)

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
        # Reshape to (batch, 1, 4, 4) if needed
        if len(x.shape) == 2:
            x = x.view(-1, 1, 4, 4)

        # Convolutional feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

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
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return preprocess_state(self.board).flatten()

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

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

        board_changed = not np.array_equal(original_board, self.board)
        return total_score, board_changed

    def get_valid_actions(self):
        """
        Return list of actions that change the board.
        Optimized to check without actually modifying the board.
        """
        valid = []
        for action in range(4):
            # Quick check: rotate, slide one row, see if it changes
            if self._would_move_change_board(action):
                valid.append(action)
        return valid

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

    def step(self, action):
        score, board_changed = self.move_board(action)

        # Advanced reward shaping
        reward = RewardShaper.shape_reward(self.board, score, board_changed)

        if board_changed:
            self.add_random_tile()

        done = self.is_game_over()
        next_state = preprocess_state(self.board).flatten()

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
else:
    learning_rate = 0.0001
    batch_size = 64
    memory_size = 50000
    epsilon_decay = 0.995
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01

print(f"Using device: {device}")
if USE_CONFIG:
    print(f"Config loaded: {'FAST' if FAST_MODE else 'FULL'} mode")
    print_config()

# ============================================================================
# INITIALIZE
# ============================================================================

env = Game2048()
state_dim = 16
action_dim = 4

q_network = DuelingCNNDQN(input_channels=1, output_dim=action_dim).to(device)
target_network = DuelingCNNDQN(input_channels=1, output_dim=action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = PrioritizedReplayBuffer(memory_size)

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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def choose_action(state, epsilon, valid_actions=None):
    """
    Epsilon-greedy action selection with optional action masking.
    """
    if random.random() < epsilon:
        if valid_actions and len(valid_actions) > 0:
            return random.choice(valid_actions)
        return random.randint(0, action_dim - 1)

    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = q_network(state)

    # If valid actions provided, mask invalid ones
    if valid_actions and len(valid_actions) > 0:
        q_values_np = q_values.cpu().numpy()[0]
        valid_q = {a: q_values_np[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    return q_values.argmax().item()


def train_step():
    """
    Perform one training step using prioritized experience replay.
    Supports mixed precision training for GPU acceleration.
    """
    if len(memory) < batch_size:
        return None

    batch, idxs, is_weights = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)
    is_weights = torch.FloatTensor(is_weights).to(device)

    # Use mixed precision if enabled
    if use_amp:
        with autocast():
            # Current Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Next Q values from target network
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

            # Weighted loss
            loss = (is_weights * (current_q_values - target_q_values.detach()).pow(2)).mean()

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

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # TD errors for priority updates
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()

        # Weighted loss
        loss = (is_weights * (current_q_values - target_q_values).pow(2)).mean()

        optimizer.zero_grad()
        loss.backward()
        max_grad = MAX_GRAD_NORM if USE_CONFIG else 10.0
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_grad)
        optimizer.step()

    # Update priorities in replay buffer
    for idx, error in zip(idxs, td_errors):
        memory.update(idx, error)

    return loss.item()


def evaluate(num_episodes=10, debug=False):
    """
    Evaluate the agent without exploration (epsilon=0).
    Returns average reward and average max tile achieved.
    """
    total_rewards = []
    max_tiles = []

    # Track invalid move statistics
    invalid_move_counts = []

    for ep_num in range(num_episodes):
        state = env.reset()
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
            print(f"Initial preprocessed state: {state[:4]}... (showing first 4 values)")

        while not done and moves < max_moves:
            # FIX: Get valid actions to prevent infinite loops
            valid_actions = env.get_valid_actions()

            # If no valid actions, game is stuck - end episode
            if not valid_actions:
                if debug and ep_num == 0:
                    print(f"\nMove {moves}: No valid actions - ending episode")
                done = True
                break

            # Store board before action
            board_before = env.board.copy()

            action = choose_action(state, epsilon=0, valid_actions=valid_actions)
            next_state, reward, done = env.step(action)

            # Check if board actually changed
            board_after = env.board.copy()
            board_changed = not np.array_equal(board_before, board_after)

            # Detect invalid moves (negative reward around -5)
            if reward <= -4:
                invalid_moves += 1

            if debug and ep_num == 0 and moves < 20:  # Log first 20 moves
                action_names = ['Up', 'Down', 'Left', 'Right']
                print(f"\n--- Move {moves} ---")
                print(f"Valid actions: {[action_names[a] for a in valid_actions]}")
                print(f"Chosen action: {action_names[action]} (index: {action})")
                print(f"Board BEFORE:\n{board_before}")
                print(f"Board AFTER:\n{board_after}")
                print(f"Board changed: {board_changed}")
                print(f"Reward: {reward}")
                print(f"Max tile: {np.max(board_after)}")

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

    return np.mean(total_rewards), np.mean(max_tiles), np.max(max_tiles)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    global epsilon

    # Use config values if available
    if USE_CONFIG:
        num_episodes = NUM_EPISODES
        target_update_freq = TARGET_UPDATE_FREQ
        eval_freq = EVAL_FREQ
        eval_episodes = EVAL_EPISODES
        save_freq = SAVE_FREQ
        log_freq = LOG_FREQ
    else:
        num_episodes = 5000
        target_update_freq = 10
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
        print(f"Network: Dueling CNN-DQN")
        print(f"Memory: Prioritized Experience Replay ({memory_size})")
        print(f"State: Log2 normalized")
        print(f"Rewards: Advanced shaping (empty tiles, monotonicity, smoothness)")
        print(f"{'='*60}\n")

    # Track training start time
    training_start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        moves = 0
        losses = []
        max_moves = 10000  # Safety limit to prevent infinite loops

        while not done and moves < max_moves:
            # Get valid actions to avoid wasting learning on invalid moves
            valid_actions = env.get_valid_actions()

            # If no valid actions, game should be over
            if not valid_actions:
                done = True
                break

            action = choose_action(state, epsilon, valid_actions)
            next_state, reward, done = env.step(action)

            # Store in prioritized replay buffer with initial high priority
            memory.add(1.0, (state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            moves += 1

            # Train
            loss = train_step()
            if loss is not None:
                losses.append(loss)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Logging
        max_tile = np.max(env.board)
        avg_loss = np.mean(losses) if losses else 0

        # TensorBoard logging
        writer.add_scalar("Training/Reward", total_reward, episode)
        writer.add_scalar("Training/MaxTile", max_tile, episode)
        writer.add_scalar("Training/Moves", moves, episode)
        writer.add_scalar("Training/Epsilon", epsilon, episode)
        writer.add_scalar("Training/Loss", avg_loss, episode)

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
            # Enable debug on first evaluation
            debug_mode = (episode == eval_freq)
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
                torch.save(q_network.state_dict(), f"models/best_model_{int(eval_avg_tile)}.pth")
                print(f"🎉 New best model saved! Avg tile: {eval_avg_tile:.1f}\n")

        # Periodic save
        if episode % save_freq == 0 and episode > 0:
            torch.save(q_network.state_dict(), f"models/checkpoint_ep{episode}.pth")

    # Final save
    torch.save(q_network.state_dict(), "models/final_model.pth")
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
