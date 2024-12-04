# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("minimaxAB_agent")
class ThirdAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(ThirdAgent, self).__init__()
    self.name = "ThirdAgent"

  

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    # Some simple code to help you with timing. Consider checking
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    max_time = 0.5

    # game stage
    occupied_tiles = sum(cell != 0 for row in chess_board for cell in row)
    # occupied_tiles = sum(sum(row) != 0 for row in chess_board)
    board_size = chess_board.shape[0] ** 2
    stage = self.determine_stage(occupied_tiles, board_size)

    best_move = None
    if stage == "start":
        best_move = self.monte_carlo_move(chess_board, player, opponent, simulations=10)
    elif stage in ["middle","end"]:
        # while not max time
        move = self.minimax_alpha_beta(chess_board, player, opponent)
        if move is not None:
            best_move = move
    return best_move

  def determine_stage(self, occupied_tiles, board_size):
      """
      Determines the stage of the game based on the number of occupied tiles.
      """
      return "end"
      # start of game if less than 25% of board is occupied
      if occupied_tiles < board_size * 0.20:
          return "start"
      # middle of game if more than 25% and less than 75% of board is occupied
      elif occupied_tiles < board_size * 0.60:
          return "middle"
      # end of game if more than 75% of board is occupied
      else:
          return "end"


  def monte_carlo_move(self, chess_board, player, opponent, simulations):
      """
              Uses Monte Carlo simulations to determine the best move.
              """
      valid_moves = get_valid_moves(chess_board, player)
      if not valid_moves:
          return None

      # create dictionary where key = valid move, and value = score associated to the move
      move_scores = {move: 0 for move in valid_moves}

      for move in valid_moves:
          for _ in range(simulations):
              board_copy = deepcopy(chess_board)
              execute_move(board_copy, move, player)

              # Simulate random play
              score = self.simulate_random_game(board_copy, player, opponent)
              move_scores[move] += score

      # Choose the move with the best average score
      best_move = max(move_scores, key=move_scores.get)
      return best_move

  def simulate_random_game(self, board, player, opponent):
      """
      Simulates a random game from the current board position.
      """
      current_player = opponent  # Start with the opponent
      while not check_endgame(board, player, opponent)[0]:
          #valid_moves = get_valid_moves(board, current_player)
          #if valid_moves:
              #random_move = random.choice(valid_moves)
              #execute_move(board, random_move, current_player)
          move = random_move(board, current_player)

          current_player = player if current_player == opponent else opponent

      # Evaluate the final board state
      player_score = np.sum(board == player)
      opponent_score = np.sum(board == opponent)
      return player_score - opponent_score


  def minimax_alpha_beta(self, chess_board, player, opponent):
      """
              Combines Minimax with Alpha-Beta pruning to find the best move.
              """
      valid_moves = get_valid_moves(chess_board, player)
      if not valid_moves:
          return None

      best_score = -float('inf')
      best_move = None
      alpha = -float('inf')
      beta = float('inf')

      for move in valid_moves:
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, player)

          # Perform Alpha-Beta pruning with Minimax
          score = self.minimax(board_copy, 2, False, player, opponent, alpha, beta)
          if score > best_score:
              best_score = score
              best_move = move

          alpha = max(alpha, score)
          if beta <= alpha:
              break  # Prune

      return best_move

  def minimax(self, board, depth, maximizing, player, opponent, alpha, beta):
      """
      Minimax function with Alpha-Beta Pruning.
      """
      # print("HERE IN MINIMAX()")


      if depth <= 0 or check_endgame(board, player, opponent)[0]:
          return self.evaluate_board(board, player, opponent)

      valid_moves = get_valid_moves(board, player if maximizing else opponent)
      if not valid_moves:
          # Pass turn if no moves are available
          return self.minimax(board, depth - 1, not maximizing, player, opponent, alpha, beta)

      if maximizing:
          max_eval = -float('inf')
          for move in valid_moves:
              board_copy = deepcopy(board)
              execute_move(board_copy, move, player)
              eval = self.minimax(board_copy, depth - 1, False, player, opponent, alpha, beta)
              max_eval = max(max_eval, eval)
              alpha = max(alpha, eval)
              if beta <= alpha:
                  break  # Prune
          return max_eval
      else:
          min_eval = float('inf')
          for move in valid_moves:
              board_copy = deepcopy(board)
              execute_move(board_copy, move, opponent)
              eval = self.minimax(board_copy, depth - 1, True, player, opponent, alpha, beta)
              min_eval = min(min_eval, eval)
              beta = min(beta, eval)
              if beta <= alpha:
                  break  # Prune
          return min_eval

  def evaluate_board(self, board, player, opponent):
    """
    Update the board with scores depending on the move that was executed
    """
    score = 0
    board_length = len(board[0])-1

    # Corners
    corners = [(0,0), (0,board_length), (board_length, 0), (board_length, board_length)]

    # Attribute scores for the positions

    # check if we moved to a corner
    # Check if corners are empty or accessible before penalizing adjacency
    for r, c in corners:
        if board[r, c] == player:
            score += 100000

        if board[r, c] == opponent: # elif or if??
            score -= 100000

        if board[r, c] == 0:  # Empty corner, adjacency is risky        # elif or if??
            # check if we are on a tile adjacent to a corner
            # adjacent to top left -> hardcode instead of r,c?
            if (r, c) == (0, 0):
                if board[r + 1, c] == player or board[r + 1, c + 1] == player or board[r, c + 1] == player:     # x squares are the worst
                    score -= 5000

            # adjacent to top right
            if (r, c) == (0, board_length):
                if board[r, c - 1] == player or board[r + 1, c - 1] == player or board[r + 1, c] == player:
                    score -= 5000

            # adjacent to bottom left
            if (r, c) == (board_length, 0):
                if board[r - 1, c] == player or board[r - 1, c + 1] == player or board[r, c + 1] == player:
                    score -= 5000

            # adjacent to bottom right
            if (r, c) == (board_length, board_length):
                if board[r, c - 1] == player or board[r - 1, c - 1] == player or board[r - 1, c] == player:
                    score -= 5000

        # check for next to a wall
        # left wall
        if board[r, 0] == player:
            score += 2000

        # top wall
        if board[0, c] == player:
            score += 2000

        # right wall
        if board[r, board_length] == player:
            score += 2000

        # bottom wall
        if board[board_length, c] == player:
            score += 2000


    # count how many pieces we captured
    #captured_pieces = count_capture(board, move, player)
    #score += captured_pieces*50 # or 1000

    # check what move could follow this move


    # check for opponent's mobility
    opponent_moves = get_valid_moves(board, opponent)
    score -= len(opponent_moves) * 2000

    # check opponent best score
    opponent_best_score = -float('inf')
    for op_move in opponent_moves:
        temp_board = deepcopy(board)
        execute_move(temp_board, op_move, opponent)
        op_score = count_capture(temp_board, op_move, opponent)  # Estimate opponent's potential score
        opponent_best_score = max(opponent_best_score, op_score)
    score -= opponent_best_score*500  # Penalize if the opponent gains a strong position


    return score
