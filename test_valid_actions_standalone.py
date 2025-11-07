"""
Standalone test for get_valid_actions() logic
Copies only the Game2048 class to avoid import issues.
"""
import numpy as np
import random


class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return self.board.flatten()

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
            self.board = np.rot90(self.board, -1)
            self.board = np.rot90(self.board, -1)
            self.board = np.rot90(self.board, -1)
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
            self.board = np.rot90(self.board, 1)
        elif direction == 1:  # Down
            self.board = np.rot90(self.board, 1)
            for i in range(4):
                new_row, score = self.slide_and_merge_row_left(self.board[i])
                self.board[i] = new_row
                total_score += score
            self.board = np.rot90(self.board, -1)
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
        """
        valid = []
        for action in range(4):
            if self._would_move_change_board(action):
                valid.append(action)
        return valid

    def _would_move_change_board(self, direction):
        """
        Fast check if a move would change the board without actually doing it.
        """
        # Create a temporary rotated view
        if direction == 0:  # Up
            temp_board = np.rot90(self.board, -3)
        elif direction == 1:  # Down
            temp_board = np.rot90(self.board, 1)
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


def test_valid_actions():
    """Test get_valid_actions() against actual board changes."""
    env = Game2048()
    action_names = ['Up', 'Down', 'Left', 'Right']

    print(f"{'='*80}")
    print(f"TESTING get_valid_actions() LOGIC")
    print(f"{'='*80}\n")

    total_tests = 0
    failed_tests = 0

    # Run 10 test scenarios
    for test_num in range(10):
        # Reset to get a random board state
        env.reset()

        print(f"\n{'='*80}")
        print(f"TEST {test_num + 1}")
        print(f"{'='*80}")
        print(f"Board state:")
        print(env.board)
        print()

        # Get what the function THINKS are valid actions
        reported_valid = env.get_valid_actions()
        print(f"get_valid_actions() returned: {[action_names[a] for a in reported_valid]} (indices: {reported_valid})")
        print()

        # Now TEST each action to see which ones ACTUALLY change the board
        actually_valid = []
        for action in range(4):
            board_before = env.board.copy()
            score, board_changed = env.move_board(action)
            board_after = env.board.copy()

            # Restore the board (undo the move)
            env.board = board_before.copy()

            if board_changed:
                actually_valid.append(action)

            status = "CHANGES" if board_changed else "NO CHANGE"
            in_reported = "YES" if action in reported_valid else "NO"

            print(f"  Action {action} ({action_names[action]:5s}): {status:9s} | In reported list: {in_reported}")

        print(f"\nActually valid actions: {[action_names[a] for a in actually_valid]} (indices: {actually_valid})")

        # Compare results
        reported_set = set(reported_valid)
        actual_set = set(actually_valid)

        total_tests += 1
        if reported_set == actual_set:
            print(f"\nPASS: get_valid_actions() is CORRECT")
        else:
            failed_tests += 1
            print(f"\nFAIL: get_valid_actions() is WRONG!")

            false_positives = reported_set - actual_set
            if false_positives:
                print(f"  False positives (reported valid but actually invalid): {[action_names[a] for a in false_positives]}")

            false_negatives = actual_set - reported_set
            if false_negatives:
                print(f"  False negatives (actually valid but not reported): {[action_names[a] for a in false_negatives]}")

    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_tests - failed_tests}")
    print(f"Failed: {failed_tests}")
    if failed_tests == 0:
        print(f"\nRESULT: ALL TESTS PASSED!")
    else:
        print(f"\nRESULT: {failed_tests} TEST(S) FAILED!")
    print(f"{'='*80}\n")


def test_specific_scenario():
    """Test a specific problematic board state."""
    env = Game2048()

    print(f"\n{'='*80}")
    print(f"SPECIFIC SCENARIO TEST")
    print(f"{'='*80}\n")

    # Create a specific board state
    # Board with only two tiles in the left column
    env.board = np.array([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    print(f"Testing board:")
    print(env.board)
    print()

    action_names = ['Up', 'Down', 'Left', 'Right']

    # Expected results:
    # Up: Should merge the two 2's into a 4 at top - VALID
    # Down: Should merge the two 2's into a 4 at bottom - VALID
    # Left: No change (already at left) - INVALID
    # Right: Should move to right - VALID

    reported_valid = env.get_valid_actions()
    print(f"get_valid_actions() says: {[action_names[a] for a in reported_valid]}")
    print()

    # Test each manually
    for action in range(4):
        board_before = env.board.copy()
        score, board_changed = env.move_board(action)
        board_after = env.board.copy()

        print(f"\nAction: {action_names[action]}")
        print(f"Board after:")
        print(board_after)
        print(f"Changed: {board_changed}")
        print(f"Score: {score}")

        # Restore
        env.board = board_before.copy()


if __name__ == "__main__":
    test_valid_actions()
    test_specific_scenario()
