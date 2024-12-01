# Student agent: Add your own agent here
from pkg_resources import non_empty_lines

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("second_agent")
class SecondAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(SecondAgent, self).__init__()
    self.name = "SecondAgent"

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

    board_length = len(chess_board)-1
    valid_moves = get_valid_moves(chess_board)

    if not valid_moves:
      # Returning a random valid move as an example
      return random_move(chess_board, player)

    # initialize
    best_move = None
    best_score = -float('inf') # worst possible score
    for move in valid_moves:
      board_copy = deepcopy(chess_board)
      execute_move(board_copy, move, player)

      # Now, we check our values to know the strength of the move
      score = self.board_score(board_copy, player, opponent, board_length, move)

      if time.time() - start_time > 1.95:
        break

      if score > best_score:
        best_move = move
        best_score = score

    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    if best_move is None:
      # Returning a random valid move as an example
      return random_move(chess_board, player)

    return best_move

  def board_score(self, board, player, opponent, board_length, move):
    """
    Update the board with scores depending on the move that was executed
    """
    score = 0

    # Corners
    corners = [(0,0), (0,board_length), (board_length, 0), (board_length, board_length)]

    # Attribute scores for the positions

    # check if we moved to a corner
    for r,c in corners:
      if board[r,c] == player:
        score += 10000
      elif board[r,c] == opponent:
        score -= 10000

    # check if we are on a tile adjacent to a corner
      # adjacent to top left -> hardcode instead of r,c?
      if (r,c) == (0,0):
        if board[r+1,c] == player or board[r+1,c+1] == player or board[r,c+1] == player:
          score -= 5000

      # adjacent to top right
      if (r,c) == (0,board_length):
        if board[r,c-1] == player or board[r+1,c-1] == player or board[r+1,c] == player:
          score -= 5000

      # adjacent to bottom left
      if (r,c) == (board_length,0):
        if board[r-1,c] == player or board[r-1,c+1] == player or board[r,c+1] == player:
          score -= 5000

      # adjacent to bottom right
      if (r,c) == (board_length,board_length):
        if board[r,c-1] == player or board[r-1,c-1] == player or board[r-1,c] == player:
          score -= 5000

      # check for next to a wall
      # left
      if c == 0:
        if board[r,c] == player:
          score += 2000

      # top
      if r == 0:
        if board[r,c] == player:
          score += 2000

      # right
      if c == board_length:
        if board[r, c] == player:
            score += 2000

      # bottom
      if r == board_length:
        if board[r,c] == player:
          score += 2000

    # count how many pieces we captured
    captured_pieces = count_capture(board, move, player)
    score += captured_pieces*100

    # check what move could follow this move


    # check for opponent's mobility
    opponent_moves = get_valid_moves(board, opponent)
    score -= len(opponent_moves) * 5

    return score




