# Reward Shaping Best Practices Research

## Phase 2: Pattern Analysis - Finding Working Examples

### Standard RL Reward Shaping Principles

1. **Reward Normalization (Industry Standard)**
   - Keep rewards in narrow range (e.g., -1 to +1 or 0 to 1)
   - Prevents Q-value explosion/collapse
   - Enables stable learning across episodes

2. **Avoid Sparse Rewards with Huge Bonuses**
   - Problem: Agent gets 0.01 for 1000 steps, then +1000 bonus
   - This is exactly what we have: ~50 for normal, +10,000 for 2048
   - Solution: Scale bonuses proportionally to normal rewards

3. **Log Scaling for Exponential Growth**
   - 2048 game has exponential tile values (2, 4, 8, 16, ..., 2048)
   - Linear bonuses (20, 100, 500, 2000, 10000) don't match
   - Should use log2(tile) for consistency

### Common Approaches in 2048 AI

**Approach 1: Score-Only (Simplest)**
- Reward = merge score only
- No bonuses
- Pros: Simple, stable
- Cons: Slower learning

**Approach 2: Log-Scaled Bonuses**
- Reward = score + log2(max_tile) * scale_factor
- Keeps bonuses proportional
- Example: log2(256) = 8, log2(1024) = 10 (only 2x difference)

**Approach 3: Normalized Composite**
- Calculate multiple components
- Normalize each to [0,1] range
- Weight and sum
- Example: reward = 0.5 * norm(score) + 0.3 * norm(empty) + 0.2 * norm(monotonicity)

**Approach 4: Reward Clipping**
- Allow varied rewards but clip to range
- Example: reward = np.clip(raw_reward, -10, 10)
- Prevents extreme values from destabilizing learning

## Current Implementation Problems

### Our Bonuses (BROKEN):
```python
if max_tile >= 2048:
    reward += 10000  # 2500x normal reward
elif max_tile >= 1024:
    reward += 2000   # 500x normal reward
elif max_tile >= 512:
    reward += 500    # 125x normal reward
elif max_tile >= 256:
    reward += 100    # 25x normal reward
elif max_tile >= 128:
    reward += 20     # 5x normal reward
```

### Why This Fails:
1. **Exponential bonuses** (20 → 100 → 500 → 2000 → 10000)
2. **Not proportional** to tile values (which double: 128 → 256 → 512 → 1024 → 2048)
3. **Massive variance** destabilizes Q-learning

## Recommended Fix

### Option 1: Log-Scaled (Best for Learning Stability)
```python
# Bonus proportional to log2(tile)
max_tile = np.max(board)
if max_tile > 0:
    tile_bonus = np.log2(max_tile) * 2  # Scale factor of 2
    # log2(128) = 7 → bonus = 14
    # log2(256) = 8 → bonus = 16
    # log2(512) = 9 → bonus = 18
    # log2(1024) = 10 → bonus = 20
    # log2(2048) = 11 → bonus = 22
    # Only 1.5x difference from 128 to 2048!
```

### Option 2: Moderate Linear (Middle Ground)
```python
# Much smaller, linear bonuses
if max_tile >= 2048:
    reward += 50    # vs 10000 before
elif max_tile >= 1024:
    reward += 40
elif max_tile >= 512:
    reward += 30
elif max_tile >= 256:
    reward += 20
elif max_tile >= 128:
    reward += 10
# Only 5x difference from 128 to 2048
```

### Option 3: Score-Only (Simplest)
```python
# Just use merge score, no bonuses
reward = score + empty_count * 2
# Remove all max_tile bonuses
# Pros: Most stable
# Cons: Slower to learn long-term strategy
```

## Hypothesis for Testing

**Hypothesis:**
Replacing exponential max_tile bonuses (20-10000) with log-scaled bonuses (14-22) will stabilize Q-learning and eliminate catastrophic forgetting.

**Expected Outcome:**
- Evaluation scores should be monotonically increasing (or stable)
- No more negative evaluation scores
- Consistent improvement toward higher tiles
- Loss should converge instead of oscillating

## Next: Phase 3 (Test Hypothesis)

We should implement ONE of these options and test if it resolves the instability.
