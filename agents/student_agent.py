# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (black),
          and 2 represents Player 2's discs (white).
        - player: 1 if this agent is playing as Player 1 (black), or 2 if playing as Player 2 (white).
        - opponent: 1 if the opponent is Player 1 (black), or 2 if the opponent is Player 2 (white).

        You should return a tuple (x, y), where (x, y) is the position where your agent
        wants to place the next disc.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # Dummy return (you should replace this with your actual logic)
        # Returning a random valid move as an example
        board_size = chess_board.shape[0]
        valid_moves = [(r, c) for r in range(board_size) for c in range(board_size) if chess_board[r, c] == 0]
        return valid_moves[0] if valid_moves else None
