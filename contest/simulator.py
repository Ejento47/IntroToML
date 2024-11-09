import numpy as np
from game_utils import is_valid_col_id, get_valid_col_id, initialize, step, is_end, is_win
from AIAgent import AIAgent
from connect_four import ConnectFour

class GameController(object):
    def __init__(self, board, agents):
        self.board = board
        self.agents_lookup = {i + 1: a for i, a in enumerate(agents)}
    def show_message(self, text):
        print(text)
        
    def draw_board(self):
        print(self.board.get_state())

    def run(self):
        self.draw_board()
        
        # Start with P1
        player_id = 1
        turn = 0
        winner_id = None  # To store the winner ID
        is_quit = False
        
        while (not is_quit) and (not self.board.is_end()):
            player_id = turn % 2 + 1
            agent = self.agents_lookup[player_id]
            try:
                action = agent.make_move(self.board.get_state())

                if action!=None:
                    
                    if action == -1:
                        is_quit = True
                        continue
                        
                    self.board.step((action, player_id))
                    self.draw_board()
                    
                    if self.board.is_win():
                        self.show_message(f"Player {player_id} wins!")
                        winner_id = player_id
                    
                    turn += 1
                    
            except ValueError as e:
                import traceback
                print(traceback.format_exc())
                self.show_message("Invalid Action!")
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.show_message("Fatal Error!")
        
        print("Actions:", self.board.get_ledger_actions())
        return winner_id

class Agent(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        self.player_id = player_id
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        return 0

# -1 is only implemented for human agents
class HumanAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)
    def make_move(self, state):
        col_id = input(f"[Player {self.player_id}] (-1 to quit) Drop Piece to: ")
        return int(col_id)

import numpy as np
from game_utils import step, get_valid_col_id, is_end
class AIAgent1(object):
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
            score += 5
        elif piece_count == 2 and empty_count == 2:
            score += 2
            
        if opp_piece_count == 3 and empty_count == 1: # Next turn opponent wins
            score -= 4
            
        return score
    
    def evaluate_board(self, board):
        """Evaluates the board state and returns a score."""
        score = 0
        #centre pref
        center_array = board[:, self.col//2]
        center_count = np.sum(center_array == self.player_id)
        score += center_count * 3
        
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
            return self.evaluate_board(state)
        
        moves = get_valid_col_id(state)
        
        if maximizing_player:
            max_eval = -float("inf")
            for col in moves:
                temp_board = step(state, col, self.player_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, False)  
                max_eval = max(max_eval, v)
                alpha = max(alpha, v)
                if max_eval >= beta:  
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for col in moves:
                temp_board = step(state, col, self.opponent_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, True) 
                min_eval = min(min_eval, v)
                beta = min(beta, v)
                if min_eval <= alpha:  
                    break
            return min_eval

    def make_move(self, state):
        best_score = -float("inf")
        best_move = None
        moves = get_valid_col_id(state)
        for move in moves:
            temp_state = step(state, int(move), self.player_id, False)
            score = self.minimax(temp_state, self.depth - 1, -float("inf"), float("inf"), False) #maximizing player
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    
class AIAgent2(object):
    def __init__(self, player_id=1, depth=3):
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
            
    def evaluate_window(self, window):
        """Evaluates a 4-cell window and returns a score based on the contents."""
        score = 0
        piece_count = np.sum(window == self.player_id)
        opp_piece_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)

        if piece_count == 4:
            score += 100
        elif piece_count == 3 and empty_count == 1:
            score += 5
        elif piece_count == 2 and empty_count == 2:
            score += 2
            
        return score
    
    def evaluate_board(self, board):
        """Evaluates the board state and returns a score."""
        score = 0
        center_array = board[:, self.col//2]
        center_count = np.sum(center_array == self.player_id)
        score += center_count * 3
        

        # Score horizontal, vertical, and diagonal windows
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                window = board[row, col:col + 4]
                score += self.evaluate_window(window)

        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 3):
                window = board[row:row + 4, col]  
                score += self.evaluate_window(window)

        # Diagonal (bottom-left to top-right)
        for row in range(board.shape[0] - 3): #row 0 to 2
            for col in range(board.shape[1] - 3): # col 0 to 3
                window = np.array([board[row + i][col + i] for i in range(4)])
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
            return self.evaluate_board(state)
        
        if maximizing_player:
            max_eval = -float("inf")
            for col in get_valid_col_id(state):
                temp_board = step(state, col, self.player_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, False)  
                max_eval = max(max_eval, v)
                alpha = max(alpha, v)
                if max_eval >= beta:  
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for col in get_valid_col_id(state):
                temp_board = step(state, col, self.opponent_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, True) 
                min_eval = min(min_eval, v)
                beta = min(beta, v)
                if min_eval <= alpha:  
                    break
            return min_eval

    def make_move(self, state):
        best_score = -float("inf")
        best_move = None
        moves = get_valid_col_id(state)
        for move in moves:
            temp_state = step(state, int(move), self.player_id, False)
            score = self.minimax(temp_state, self.depth - 1, -float("inf"), float("inf"), False) #maximizing player
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

class AIAgent3(object):
    def __init__(self, player_id=1, depth=3):
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
            
    def evaluate_window(self, window):
        """Evaluates a 4-cell window and returns a score based on the contents."""
        score = 0
        piece_count = np.sum(window == self.player_id)
        opp_piece_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)

        if piece_count == 4: # player win unnecesary but i just keptit
            score += 1000
        elif piece_count == 3 and empty_count == 1: # Close to winning
            score += 10
        elif piece_count == 2 and empty_count == 2: # Potential to win
            score += 5
            
        if opp_piece_count == 3 and empty_count == 1: # Next turn opponent wins harshppunish
            score -= 100
        if opp_piece_count == 2 and empty_count == 2:
            score -= 5
            
        return score
    
    def evaluate_board(self, board):
        """Evaluates the board state and returns a score."""
        score = 0
        center_array = board[:, self.col//2]
        center_count = np.sum(center_array == self.player_id)
        score += center_count * 3
        

        # Score horizontal, vertical, and diagonal windows
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                window = board[row, col:col + 4]
                score += self.evaluate_window(window)

        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 3):
                window = board[row:row + 4, col]  
                score += self.evaluate_window(window)

        # Diagonal (bottom-left to top-right)
        for row in range(board.shape[0] - 3): #row 0 to 2
            for col in range(board.shape[1] - 3): # col 0 to 3
                window = np.array([board[row + i][col + i] for i in range(4)])
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
            return self.evaluate_board(state)
        
        if maximizing_player:
            max_eval = -float("inf")
            for col in get_valid_col_id(state):
                temp_board = step(state, col, self.player_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, False)  
                max_eval = max(max_eval, v)
                alpha = max(alpha, v)
                if max_eval >= beta:  
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for col in get_valid_col_id(state):
                temp_board = step(state, col, self.opponent_id, False)  
                v = self.minimax(temp_board, depth - 1, alpha, beta, True) 
                min_eval = min(min_eval, v)
                beta = min(beta, v)
                if min_eval <= alpha:  
                    break
            return min_eval

    def make_move(self, state):
        best_score = -float("inf")
        best_move = None
        moves = get_valid_col_id(state)
        for move in moves:
            temp_state = step(state, int(move), self.player_id, False)
            score = self.minimax(temp_state, self.depth - 1, -float("inf"), float("inf"), False) #maximizing player
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    
if __name__ == "__main__":
    #compete against aiagent
    a_ = 0
    a1 = 0
    a2 = 0
    a3 = 0

    for i in range(30):
        board = ConnectFour()
        if i < 5:
            game = GameController(board=board, agents=[AIAgent(1), AIAgent1(2)])
            winner_id = game.run()
            if winner_id == 1:
                a1 += 1
            if winner_id == 2:
                a_ += 1

        if i >= 5 and i < 10:
            game = GameController(board=board, agents=[AIAgent2(1), AIAgent(2)])
            winner_id = game.run()
            if winner_id == 2:
                a1 += 1
            if winner_id == 1:
                a_ += 1
        if i >= 10 and i < 15:
            game = GameController(board=board, agents=[AIAgent(1), AIAgent2(2)])
            winner_id = game.run()
            if winner_id == 1:
                a2 += 1
            if  winner_id == 2:
                a_ += 1
        if i >= 15 and i < 20:
            game = GameController(board=board, agents=[AIAgent2(1), AIAgent(2)])
            winner_id = game.run()
            if winner_id == 1:
                a_ += 1
            if winner_id == 2:
                a2 += 1
        if i >= 20 and i < 25:
            game = GameController(board=board, agents=[AIAgent(1), AIAgent3(2)])
            winner_id = game.run()
            if winner_id == 1:
                a3 += 1
            if winner_id == 2:
                a_ += 1
        if i >= 25:
            game = GameController(board=board, agents=[AIAgent3(1), AIAgent(2)])
            winner_id = game.run()
            if winner_id == 1:
                a_ += 1
            if winner_id == 2:
                a3 += 1
    print("A1 vs A")
    print(a1)
    print("A2 vs A")
    print(a2)
    print("A vs A3")
    print(a3)
    print("A loss")
    print(a_)
        

    # # game = GameControllerPygame(board=board, agents=[AIAgent2(1), AIAgent2(2)])
    # #create multiple games and take the average of the winners between agents 1, 2, and 3
    # avsa2 = 0
    # a2vsa = 0
    
    # avsa3 = 0
    # a3vsa = 0
    
    # a2vsa3 = 0
    # a3vsa2 = 0
    # for i in range(30):
    #     board = ConnectFour()
    #     if i < 5:
            
    #         game = GameController(board=board, agents=[AIAgent1(1), AIAgent2(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             avsa2 += 1
    #         if winner_id == 2:
    #             a2vsa += 1
    #     if i >= 5 and i < 10:
    #         game = GameController(board=board, agents=[AIAgent2(1), AIAgent1(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             a2vsa += 1
    #         if winner_id == 2:
    #             avsa2 += 1
    #     if i >= 10 and i < 15:
    #         game = GameController(board=board, agents=[AIAgent1(1), AIAgent3(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             avsa3 += 1
    #         if winner_id == 2:
    #             a3vsa += 1
    #     if i >= 15 and i < 20:
    #         game = GameController(board=board, agents=[AIAgent3(1), AIAgent1(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             a3vsa += 1
    #         if winner_id == 2:
    #             avsa3 += 1  
    #     if  i >= 20 and i < 25:
    #         game = GameController(board=board, agents=[AIAgent2(1), AIAgent3(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             a2vsa3 += 1
    #         if winner_id == 2:
    #             a3vsa2 += 1
    #     if i >= 25:
    #         game = GameController(board=board, agents=[AIAgent3(1), AIAgent2(2)])
    #         winner_id = game.run()
    #         if winner_id == 1:
    #             a3vsa2 += 1
    #         if winner_id == 2:
    #             a2vsa3 += 1

    # print("A vs A2")
    # print(avsa2)
    # print(a2vsa)
    # print("A vs A3")
    # print(avsa3)
    # print(a3vsa)
    # print("A2 vs A3")
    # print(a2vsa3)
    # print(a3vsa2)