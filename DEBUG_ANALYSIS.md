# Root Cause Analysis - Evaluation Failure

## The Smoking Gun

**Observation from output:**
```
Ep 400: Avg Reward: -49954.4 | Avg Max Tile: 3.3 | Best: 4
```

Reward of **-49,954** with max tile of only **4** (2^2).

## Calculation Breakdown

Given the reward function:
- Invalid move penalty: **-5**
- Max move limit in evaluation: **10,000 moves**

If the AI makes mostly invalid moves:
```
10,000 moves × -5 penalty = -50,000 reward
```

**This exactly matches the evaluation rewards!**

## The Critical Question

**How can the AI make invalid moves when we have action masking?**

```python
# In evaluate():
valid_actions = env.get_valid_actions()
action = choose_action(state, epsilon=0, valid_actions=valid_actions)
```

The code explicitly masks invalid actions!

## Root Cause - FOUND! ✓

**The bug was in `_would_move_change_board()` function (line 444-477)**

### The Buggy Logic:
```python
if len(non_zero) < len(row):
    return True  # Has gaps, will change  ← WRONG!
```

This incorrectly assumed that any row with gaps will change when moved.

**Counterexample:**
- Row: `[2, 0, 0, 0]`
- Moving LEFT: Still `[2, 0, 0, 0]` - NO CHANGE!
- But the function returned `True` because it saw gaps

### The Fix:
```python
# Check if tiles are already in leftmost positions
for i in range(len(non_zero)):
    if row[i] != non_zero[i]:
        return True  # Tiles would move to different positions
```

Now correctly checks if non-zero elements are already in their final positions.

## Test Results

**Before fix:** 2/10 tests failed (false positives)
**After fix:** 10/10 tests passed ✓

**Example:**
- Board: `[[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]`
- Before: `get_valid_actions()` returned `['Up', 'Down', 'Left', 'Right']` ❌
- After: `get_valid_actions()` returns `['Up', 'Down', 'Right']` ✓ (Left correctly excluded)

## Impact

The AI was selecting "valid" actions that were actually invalid, receiving -5 penalty repeatedly:
- ~10,000 invalid moves per episode
- Total reward: ~-50,000 per episode
- Max tile achieved: Only 4 (barely any valid moves made)

**Status: FIXED in main_advanced.py:444-477**
