import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


def preprocess_state(board):
    """One-hot encode a 4x4 board into (16, 4, 4)."""
    encoded = np.zeros((16, 4, 4), dtype=np.float32)
    board_int = board.astype(np.int64, copy=False)
    planes = np.zeros_like(board_int, dtype=np.int64)
    non_zero = board_int > 0
    if np.any(non_zero):
        planes[non_zero] = np.clip(np.log2(board_int[non_zero]).astype(np.int64), 1, 15)
    row_idx, col_idx = np.indices(board_int.shape)
    encoded[planes, row_idx, col_idx] = 1.0
    return encoded


class Game2048:
    """
    2048 environment for RL training.
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed) if seed is not None else random
        self.board = np.zeros((4, 4), dtype=int)
        self.moves = 0
        self.max_moves = 10000
        self._valid_actions_cache = None
        self._valid_action_mask_cache = None
        self.reset()

    def _invalidate_valid_actions_cache(self):
        self._valid_actions_cache = None
        self._valid_action_mask_cache = None

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.moves = 0
        self.add_random_tile()
        self.add_random_tile()
        self._invalidate_valid_actions_cache()
        return preprocess_state(self.board)

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = self.rng.choice(empty_cells)
            self.board[row, col] = 2 if self.rng.random() < 0.9 else 4
            self._invalidate_valid_actions_cache()

    @staticmethod
    def slide_and_merge_row_left(row):
        non_zero = row[row != 0]
        if len(non_zero) == 0:
            return row.copy(), 0

        score = 0
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                value = non_zero[i] * 2
                merged.append(value)
                score += value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        merged.extend([0] * (len(row) - len(merged)))
        return np.array(merged, dtype=int), score

    def move_board(self, direction):
        original = self.board.copy()
        total_score = 0

        if direction == 0:  # up
            board = np.rot90(self.board, 1)
            for i in range(4):
                board[i], s = self.slide_and_merge_row_left(board[i])
                total_score += s
            self.board = np.rot90(board, -1)
        elif direction == 1:  # down
            board = np.rot90(self.board, -1)
            for i in range(4):
                board[i], s = self.slide_and_merge_row_left(board[i])
                total_score += s
            self.board = np.rot90(board, 1)
        elif direction == 2:  # left
            board = self.board.copy()
            for i in range(4):
                board[i], s = self.slide_and_merge_row_left(board[i])
                total_score += s
            self.board = board
        elif direction == 3:  # right
            board = np.rot90(self.board, 2)
            for i in range(4):
                board[i], s = self.slide_and_merge_row_left(board[i])
                total_score += s
            self.board = np.rot90(board, 2)
        else:
            raise ValueError(f"Invalid action {direction}")

        changed = not np.array_equal(original, self.board)
        if changed:
            self._invalidate_valid_actions_cache()
        return total_score, changed

    def _would_move_change_board(self, direction):
        if direction == 0:
            temp = np.rot90(self.board, 1)
        elif direction == 1:
            temp = np.rot90(self.board, -1)
        elif direction == 2:
            temp = self.board
        elif direction == 3:
            temp = np.rot90(self.board, 2)
        else:
            return False

        for row in temp:
            non_zero = row[row != 0]
            if len(non_zero) == 0:
                continue
            for i in range(len(non_zero) - 1):
                if non_zero[i] == non_zero[i + 1]:
                    return True
            for i in range(len(non_zero)):
                if row[i] != non_zero[i]:
                    return True
        return False

    def get_valid_actions(self):
        if self._valid_actions_cache is None:
            mask = np.zeros(4, dtype=bool)
            for a in range(4):
                mask[a] = self._would_move_change_board(a)
            self._valid_action_mask_cache = mask
            self._valid_actions_cache = np.flatnonzero(mask).tolist()
        return list(self._valid_actions_cache)

    def get_valid_action_mask(self):
        if self._valid_action_mask_cache is None:
            self.get_valid_actions()
        return self._valid_action_mask_cache.copy()

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return False
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return False
        return True

    def step(self, action):
        score, changed = self.move_board(action)
        if changed:
            self.add_random_tile()
            reward = float(np.log1p(score))
        else:
            reward = -1.0

        self.moves += 1
        done = self.is_game_over() or self.moves >= self.max_moves
        valid_actions = [] if done else self.get_valid_actions()
        valid_mask = np.zeros(4, dtype=bool) if done else self.get_valid_action_mask()
        next_state = preprocess_state(self.board)

        info = {
            "board_changed": changed,
            "max_tile": int(np.max(self.board)),
            "valid_actions": valid_actions,
            "valid_action_mask": valid_mask,
            "score": score,
        }
        return next_state, reward, done, info


class VecGame2048:
    """Simple synchronous vectorized 2048 env."""

    def __init__(self, n_envs, base_seed=42):
        self.envs = [Game2048(seed=base_seed + i) for i in range(n_envs)]
        self.n_envs = n_envs

    def reset(self):
        obs = []
        masks = []
        for env in self.envs:
            obs.append(env.reset())
            masks.append(env.get_valid_action_mask())
        return np.stack(obs), np.stack(masks)

    def step(self, actions):
        next_obs = []
        rewards = []
        dones = []
        infos = []
        next_masks = []

        for i, env in enumerate(self.envs):
            obs, rew, done, info = env.step(int(actions[i]))
            if done:
                obs = env.reset()
                info["terminal"] = True
            else:
                info["terminal"] = False
            next_obs.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
            next_masks.append(env.get_valid_action_mask())

        return (
            np.stack(next_obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
            np.stack(next_masks),
        )


class ActorCriticNet(nn.Module):
    def __init__(self, input_channels=16, n_actions=4, conv_filters=128, hidden_size=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(conv_filters, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(conv_filters, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        z = self.features(x).flatten(1)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    total_timesteps: int
    n_envs: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef_start: float
    ent_coef_end: float
    vf_coef: float
    max_grad_norm: float
    learning_rate_start: float
    learning_rate_end: float
    eval_every_updates: int
    eval_episodes: int
    save_every_updates: int
    conv_filters: int
    hidden_size: int
    use_compile: bool
    seed: int


def get_default_config(colab_profile):
    if colab_profile:
        return PPOConfig(
            total_timesteps=4_000_000,
            n_envs=32,
            rollout_steps=128,
            update_epochs=4,
            minibatch_size=1024,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef_start=0.01,
            ent_coef_end=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate_start=3e-4,
            learning_rate_end=5e-5,
            eval_every_updates=20,
            eval_episodes=20,
            save_every_updates=50,
            conv_filters=128,
            hidden_size=256,
            use_compile=True,
            seed=42,
        )
    return PPOConfig(
        total_timesteps=1_500_000,
        n_envs=8,
        rollout_steps=128,
        update_epochs=4,
        minibatch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef_start=0.01,
        ent_coef_end=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate_start=3e-4,
        learning_rate_end=5e-5,
        eval_every_updates=20,
        eval_episodes=10,
        save_every_updates=50,
        conv_filters=96,
        hidden_size=192,
        use_compile=False,
        seed=42,
    )


def apply_action_mask(logits, valid_mask):
    mask_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~valid_mask, mask_value)


def linear_schedule(start_value, end_value, progress):
    """Linear interpolation from start to end, progress in [0, 1]."""
    p = float(np.clip(progress, 0.0, 1.0))
    return start_value + p * (end_value - start_value)


def evaluate_policy(model, device, n_episodes=10, seed=10_000):
    model.eval()
    rewards = []
    max_tiles = []

    with torch.inference_mode():
        for ep in range(n_episodes):
            env = Game2048(seed=seed + ep)
            obs = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                mask_np = env.get_valid_action_mask()
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device, dtype=torch.bool)
                logits, _ = model(obs_t)
                logits = apply_action_mask(logits, mask_t)
                action = logits.argmax(dim=1).item()
                obs, rew, done, _ = env.step(action)
                total_reward += rew

            rewards.append(total_reward)
            max_tiles.append(int(np.max(env.board)))

    model.train()
    max_tiles_np = np.array(max_tiles, dtype=np.int32)
    metrics = {
        "rate_256": float(np.mean(max_tiles_np >= 256)),
        "rate_512": float(np.mean(max_tiles_np >= 512)),
        "rate_1024": float(np.mean(max_tiles_np >= 1024)),
        "rate_2048": float(np.mean(max_tiles_np >= 2048)),
    }

    return float(np.mean(rewards)), float(np.mean(max_tiles)), int(np.max(max_tiles)), metrics


def train_ppo(args):
    colab_profile = "COLAB_GPU" in os.environ and torch.cuda.is_available()
    cfg = get_default_config(colab_profile=colab_profile)

    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    run_name = f"ppo_2048_{'colab' if colab_profile else 'local'}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    print("=" * 72)
    print("PPO 2048 TRAINING")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Colab profile: {colab_profile}")
    print(f"Total timesteps: {cfg.total_timesteps:,}")
    print(f"Envs: {cfg.n_envs} | Rollout steps: {cfg.rollout_steps} | Batch/update: {cfg.n_envs * cfg.rollout_steps}")
    print(
        f"LR: {cfg.learning_rate_start} -> {cfg.learning_rate_end} | "
        f"Entropy: {cfg.ent_coef_start} -> {cfg.ent_coef_end} | "
        f"Epochs: {cfg.update_epochs} | Minibatch: {cfg.minibatch_size}"
    )
    print("=" * 72)

    envs = VecGame2048(cfg.n_envs, base_seed=cfg.seed)
    model = ActorCriticNet(
        n_actions=4,
        conv_filters=cfg.conv_filters,
        hidden_size=cfg.hidden_size,
    ).to(device)

    if cfg.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="default")
            print("torch.compile: ENABLED")
        except Exception as exc:
            print(f"torch.compile: DISABLED ({exc})")
    else:
        print("torch.compile: DISABLED")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate_start, eps=1e-5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    obs_np, masks_np = envs.reset()
    obs = torch.from_numpy(obs_np).to(device)
    valid_masks = torch.from_numpy(masks_np).to(device, dtype=torch.bool)

    batch_size = cfg.n_envs * cfg.rollout_steps
    n_updates = cfg.total_timesteps // batch_size
    best_eval_tile = 0.0
    start_update = 1
    steps_done = 0

    start = time.time()

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_to_save.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_update = int(checkpoint.get("update", 0)) + 1
            best_eval_tile = float(checkpoint.get("best_eval_tile", 0.0))
            steps_done = int(checkpoint.get("steps_done", 0))
            print(f"Resumed from {args.resume} at update {start_update - 1}, steps {steps_done}.")
        else:
            # Backward compatibility: load model-only checkpoint
            model_to_save.load_state_dict(checkpoint)
            print(f"Loaded model-only checkpoint from {args.resume}. Optimizer state not restored.")

    for update in range(start_update, n_updates + 1):
        update_progress = update / max(n_updates, 1)
        current_lr = linear_schedule(cfg.learning_rate_start, cfg.learning_rate_end, update_progress)
        current_ent_coef = linear_schedule(cfg.ent_coef_start, cfg.ent_coef_end, update_progress)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        obs_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs, 16, 4, 4), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs), dtype=torch.long, device=device)
        logprobs_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs), dtype=torch.float32, device=device)
        values_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs), dtype=torch.float32, device=device)
        masks_buf = torch.zeros((cfg.rollout_steps, cfg.n_envs, 4), dtype=torch.bool, device=device)
        max_tile_ep = []

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            masks_buf[t] = valid_masks

            with torch.inference_mode():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, values = model(obs)
                else:
                    logits, values = model(obs)
                logits = apply_action_mask(logits, valid_masks)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_np = actions.detach().cpu().numpy()
            next_obs_np, rewards_np, dones_np, infos, next_masks_np = envs.step(actions_np)

            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            values_buf[t] = values
            rewards_buf[t] = torch.from_numpy(rewards_np).to(device)
            dones_buf[t] = torch.from_numpy(dones_np).to(device)

            for i, info in enumerate(infos):
                if info.get("terminal", False):
                    max_tile_ep.append(info["max_tile"])

            obs = torch.from_numpy(next_obs_np).to(device)
            valid_masks = torch.from_numpy(next_masks_np).to(device, dtype=torch.bool)

        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, next_value = model(obs)
            else:
                _, next_value = model(obs)

        advantages = torch.zeros_like(rewards_buf, device=device)
        last_gae = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)

        for t in reversed(range(cfg.rollout_steps)):
            non_terminal = 1.0 - dones_buf[t]
            delta = rewards_buf[t] + cfg.gamma * next_value * non_terminal - values_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = values_buf[t]

        returns = advantages + values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_obs = obs_buf.reshape(-1, 16, 4, 4)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape(-1, 4)

        idxs = np.arange(batch_size)
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start_idx in range(0, batch_size, cfg.minibatch_size):
                mb_idx = idxs[start_idx:start_idx + cfg.minibatch_size]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprobs = b_logprobs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_old_values = b_values[mb_idx]
                mb_masks = b_masks[mb_idx]

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, values = model(mb_obs)
                        logits = apply_action_mask(logits, mb_masks)
                        dist = Categorical(logits=logits)
                        new_logprobs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_logprobs - mb_old_logprobs)
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                        value_pred_clipped = mb_old_values + (values - mb_old_values).clamp(-cfg.clip_coef, cfg.clip_coef)
                        value_loss_1 = (values - mb_returns).pow(2)
                        value_loss_2 = (value_pred_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                        loss = policy_loss + cfg.vf_coef * value_loss - current_ent_coef * entropy

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, values = model(mb_obs)
                    logits = apply_action_mask(logits, mb_masks)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - mb_old_logprobs)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_pred_clipped = mb_old_values + (values - mb_old_values).clamp(-cfg.clip_coef, cfg.clip_coef)
                    value_loss_1 = (values - mb_returns).pow(2)
                    value_loss_2 = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                    loss = policy_loss + cfg.vf_coef * value_loss - current_ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()

        steps_done = update * batch_size
        elapsed = time.time() - start
        sps = int(steps_done / max(elapsed, 1e-6))

        if update % 10 == 0 or update == 1:
            max_tile_train = float(np.mean(max_tile_ep)) if max_tile_ep else float(np.max([np.max(e.board) for e in envs.envs]))
            print(
                f"Update {update:4d}/{n_updates} | Steps: {steps_done:>9d} | SPS: {sps:>6d} | "
                f"PolicyLoss: {policy_loss_epoch:.3f} | ValueLoss: {value_loss_epoch:.3f} | "
                f"Entropy: {entropy_epoch:.3f} | LR: {current_lr:.2e} | EntCoef: {current_ent_coef:.4f} | "
                f"AvgTrainMaxTile: {max_tile_train:.1f}"
            )

        writer.add_scalar("Train/SPS", sps, update)
        writer.add_scalar("Train/PolicyLossSum", policy_loss_epoch, update)
        writer.add_scalar("Train/ValueLossSum", value_loss_epoch, update)
        writer.add_scalar("Train/EntropySum", entropy_epoch, update)
        writer.add_scalar("Train/AdvantageMean", advantages.mean().item(), update)
        writer.add_scalar("Train/LearningRate", current_lr, update)
        writer.add_scalar("Train/EntropyCoef", current_ent_coef, update)

        if update % cfg.eval_every_updates == 0:
            eval_reward, eval_avg_tile, eval_best_tile, eval_metrics = evaluate_policy(
                model=model,
                device=device,
                n_episodes=cfg.eval_episodes,
                seed=50_000 + update * 10,
            )
            writer.add_scalar("Eval/AvgReward", eval_reward, update)
            writer.add_scalar("Eval/AvgMaxTile", eval_avg_tile, update)
            writer.add_scalar("Eval/BestTile", eval_best_tile, update)
            writer.add_scalar("Eval/Rate256", eval_metrics["rate_256"], update)
            writer.add_scalar("Eval/Rate512", eval_metrics["rate_512"], update)
            writer.add_scalar("Eval/Rate1024", eval_metrics["rate_1024"], update)
            writer.add_scalar("Eval/Rate2048", eval_metrics["rate_2048"], update)

            print(
                f"\n{'=' * 60}\n"
                f"PPO EVALUATION (Update {update})\n"
                f"Avg Reward: {eval_reward:.1f} | Avg Max Tile: {eval_avg_tile:.1f} | Best: {eval_best_tile}\n"
                f"Rate>=256: {eval_metrics['rate_256']:.2%} | Rate>=512: {eval_metrics['rate_512']:.2%} | "
                f"Rate>=1024: {eval_metrics['rate_1024']:.2%} | Rate>=2048: {eval_metrics['rate_2048']:.2%}\n"
                f"{'=' * 60}\n"
            )

            if eval_avg_tile > best_eval_tile:
                best_eval_tile = eval_avg_tile
                os.makedirs("models", exist_ok=True)
                torch.save(model_to_save.state_dict(), f"models/ppo_best_model_{int(best_eval_tile)}.pth")
                print(f"New PPO best model saved: avg tile {best_eval_tile:.1f}")

        if update % cfg.save_every_updates == 0:
            os.makedirs("models", exist_ok=True)
            checkpoint = {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "steps_done": steps_done,
                "best_eval_tile": best_eval_tile,
                "config": cfg.__dict__,
            }
            torch.save(checkpoint, f"models/ppo_checkpoint_update{update}.pth")

    os.makedirs("models", exist_ok=True)
    torch.save(model_to_save.state_dict(), "models/ppo_final_model.pth")
    writer.close()
    print("PPO training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for 2048.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--resume", type=str, default=None, help="Path to PPO checkpoint for resume.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
