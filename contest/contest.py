import re
from game_utils import initialize, is_end, step, get_valid_col_id

c4_board = initialize()
print(c4_board.shape)

print(c4_board)

# get_valid_col_id(c4_board)

# step(c4_board, col_id=2, player_id=1, in_place=True)

# print(c4_board)

# step(c4_board, col_id=2, player_id=2, in_place=True)
# step(c4_board, col_id=2, player_id=1, in_place=True)
# step(c4_board, col_id=2, player_id=2, in_place=True)
# step(c4_board, col_id=2, player_id=1, in_place=True)
# step(c4_board, col_id=2, player_id=2, in_place=True)
# print(c4_board)

# print(get_valid_col_id(c4_board))

# step(c4_board, col_id=2, player_id=1, in_place=True)

class ZeroAgent(object):
    def __init__(self, player_id=1):
        pass
    def make_move(self, state):
        return 0

# Step 1
agent1 = ZeroAgent(player_id=1) # Yours, Player 1
agent2 = ZeroAgent(player_id=2) # Opponent, Player 2

# Step 2
contest_board = initialize()

# Step 3
p1_board = contest_board.view()
p1_board.flags.writeable = False
move1 = agent1.make_move(p1_board)

# Step 4
step(contest_board, col_id=move1, player_id=1, in_place=True)

from simulator import GameController, HumanAgent
from connect_four import ConnectFour
import numpy as np
import time
from game_utils import step, get_valid_col_id, is_end, is_win 

class AIAgent(object):
    """
    A class representing an AI agent that plays Connect Four using Minimax with Alpha-Beta Pruning.
    """
    def __init__(self, player_id=1, depth=3):
        """Initializes the agent with the specified player ID and search depth.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        depth : int
            The depth to search in the minimax algorithm.
        """
        self.player_id = player_id
        self.depth = depth
        self.opp = 3 - player_id
        
    def flatten_board(self, board):
        """Flattens the 2D board into a 1D array."""
        return board.flatten()

    def index_to_row_col(self, index, cols):
        """Converts a flattened index back to (row, col)."""
        row = index // cols
        col = index % cols
        return row, col

    def reorder(self, valid_cols):
        center_col = len(valid_cols) // 2
        valid_cols = sorted(valid_cols, key=lambda x: abs(center_col - x))
        return valid_cols

    def evaluate_window(self, window, player_id):
        """Evaluates a 4-cell window and returns a score based on the contents."""
        score = 0
        opponent_id = 3 - player_id
        piece_count = np.sum(window == player_id)
        opp_piece_count = np.sum(window == opponent_id)
        empty_count = np.sum(window == 0)

        # For player's potential
        if piece_count == 4:  # Instant win for player
            score += 1000
        elif piece_count == 3 and empty_count == 1:  # Close to winning
            score += 20  # Increased for higher priority
        elif piece_count == 2 and empty_count == 2:  # Potential to win
            score += 5
        elif piece_count == 1 and empty_count == 3:  # Small potential
            score += 1

        # Punish threats from opponent
        if opp_piece_count == 3 and empty_count == 1:  # Next turn opponent wins
            score -= 100
        elif opp_piece_count == 2 and empty_count == 2:
            score -= 5
        elif opp_piece_count == 1 and empty_count == 3:
            score -= 1

        return score

    def evaluate_board(self, flattened_board, rows, cols, player_id):
        """Evaluates the entire board and returns a score."""
        score = 0

        # Center column preference (heavier weight)
        center_array = flattened_board[cols // 2::cols]  # Extract center column
        center_count = np.count_nonzero(center_array == player_id)
        score += center_count * 3

        # Horizontal score (windows of 4 consecutive cells)
        for row in range(rows):
            for col in range(cols - 3):  # Ensure there's room for a window
                window = flattened_board[row * cols + col: row * cols + col + 4]
                score += self.evaluate_window(window, player_id)

        # Vertical score (windows of 4 consecutive cells)
        for col in range(cols):
            for row in range(rows - 3):  # Ensure there's room for a vertical window
                window = [flattened_board[(row + i) * cols + col] for i in range(4)]
                score += self.evaluate_window(window, player_id)

        # Diagonal left-to-right
        for row in range(rows - 3):
            for col in range(cols - 3):
                window = [
                    flattened_board[(row + i) * cols + (col + i)] for i in range(4)
                ]
                score += self.evaluate_window(window, player_id)

        # Diagonal right-to-left
        for row in range(3, rows):
            for col in range(cols - 3):
                window = [
                    flattened_board[(row - i) * cols + (col + i)] for i in range(4)
                ]
                score += self.evaluate_window(window, player_id)

        return score

    def minimax(self, state, depth, alpha, beta, player_id):
        """Minimax algorithm with Alpha-Beta Pruning."""
        rows, cols = state.shape
        flattened_board = self.flatten_board(state)

        # Terminal state check
        if is_end(state) or depth == 0:
            if player_id != self.player_id:
                return -self.evaluate_board(flattened_board, rows, cols, self.player_id)
            return self.evaluate_board(flattened_board, rows, cols, self.player_id)

        if player_id == self.player_id:
            v = -float("inf")
            cols = self.reorder(get_valid_col_id(state))
            for col in cols:
                temp_state = step(state, col, self.player_id, False)
                v = max(v, self.minimax(temp_state, depth - 1, alpha, beta, 3 - player_id))  # Opponent's turn
                alpha = max(alpha, v)
                if v >= beta:
                    return v
            return v
        else:
            v = float("inf")
            cols = self.reorder(get_valid_col_id(state))
            for col in cols:
                temp_state = step(state, col, self.opp, False)
                v = min(v, self.minimax(temp_state, depth - 1, alpha, beta, self.player_id))  # Maximizing player
                beta = min(beta, v)
                if v <= alpha:
                    return v
            return v

    def make_move(self, state):
        """
        Determines and returns the best move for the agent based on the current game state.
        """
        moves = self.reorder(get_valid_col_id(state))
        t = []

        # Evaluate all possible moves
        for move in moves:
            temp_state = step(state, move, self.player_id, False)
            flattened_state = self.flatten_board(temp_state)
            score = self.minimax(temp_state, self.depth - 1, -float("inf"), float("inf"), self.player_id)
            t.append((score, move))

        # Select the move with the highest score (random if there are multiple)
        best_score = max(t)[0]
        best_moves = [move for score, move in t if score == best_score]
        best_move = np.random.choice(best_moves)
        return best_move


board = np.array([
    [1, 2, 1, 2, 1, 0, 0],
    [2, 1, 2, 1, 2, 0, 0],
    [1, 2, 1, 2, 1, 0, 0],
    [2, 1, 2, 1, 2, 0, 0],
    [1, 2, 1, 2, 1, 0, 0],
    [2, 1, 2, 1, 2, 0, 0]
])

ai_agent = AIAgent(player_id=1, depth=3)

# AI should choose the best move to either block the opponent or win
ai_move = ai_agent.make_move(board)
print("AI's move in the endgame:", ai_move)

