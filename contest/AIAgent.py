from game_utils import initialize, is_end, is_win, step, get_valid_col_id
import numpy as np


class AIAgent(object):
    """
    A class representing an AI agent that plays Connect Four using Minimax with Alpha-Beta Pruning.
    """
    def __init__(self, player_id=1, depth=6):
        """Initializes the agent with the specified player ID and search depth.
        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        depth : int
            The depth to search in the minimax algorithm.
        time_limit : float
            The time limit in seconds for the agent to make a move.
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.depth = depth
        self.row = 6
        self.col = 7
    # def reorder(self, valid_cols):
    #     center_col = len(valid_cols) // 2
    #     valid_cols = sorted(valid_cols, key=lambda x: abs(center_col - x))
    #     return valid_cols
    
    def evaluate_window(self, window):
        """Evaluates a 4-cell window and returns a score based on the contents."""
        score = 0
        piece_count = np.sum(window == self.player_id)
        opp_piece_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)


        if piece_count == 4:
            score += 100
        elif piece_count == 3 and empty_count == 1:
            score += 10  
        elif piece_count == 2 and empty_count == 2:
            score += 3 
        

        if opp_piece_count == 3 and empty_count == 1:
            score -= 15  # harsh piunishment since next turn win
        
        return score
    
    def evaluate_board(self, board):
        """Evaluates the board state and returns a score."""
        score = 0
        #centre pref
        center_array = board[:, self.col//2]
        center_count = np.sum(center_array == self.player_id)
        score += center_count * 6
        
        # Score horizontal, vertical, and diagonal windows
        for row in range(self.row):
            for col in range(self.col - 3):
                window = board[row, col: col + 4]
                score += self.evaluate_window(window)

        for col in range(self.col):
            for row in range(self.row - 3):
                window = board[row:row + 4, col]  
                score += self.evaluate_window(window)

        # Diagonal (bottom-left to top-right)
        for row in range(board.shape[0] - 3): #row 0 to 2
            for col in range(board.shape[1] - 3): # col 0 to 3
                window = np.array([board[row + i][col + i] for i in range(4)]) #conver to arr so np.sum
                score += self.evaluate_window(window)

        # Diagonal (top-left to bottom-right)
        for row in range(board.shape[0] - 3): #row 0 to 2
            for col in range(board.shape[1] - 3): # col 0 to 3
                window = np.array([board[row + 3 - i][col + i] for i in range(4)]) #from 6,0 to 3,3
                score += self.evaluate_window(window)
        return score
    
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        # terminal state
        if is_end(state) or depth == 0:
            return (None,self.evaluate_board(state))
             
        if maximizing_player:
            score = -float("inf")
            m = None
            for move in get_valid_col_id(state):
                temp_board = step(state, move, self.player_id, False)  
                _,v = self.minimax(temp_board, depth - 1, alpha, beta, False)
                if v > score:
                    score = v
                    m = move
                alpha = max(alpha, v)
                if v >= beta:
                    return m, score
            return m, score
        else:
            score = float("inf")
            m = None
            for move in get_valid_col_id(state):
                temp_board = step(state, move, self.opponent_id, False)  
                _,v = self.minimax(temp_board, depth - 1, alpha, beta, True)  
                if v < score:
                    score = v
                    m = move
                beta = min(beta, v)
                if v <= alpha:
                    return m, score
            return m, score

    def make_move(self, state):
        move, _ = self.minimax(state, self.depth, -float("inf"), float("inf"), True)
        return move