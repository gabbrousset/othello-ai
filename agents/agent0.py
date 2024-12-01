import copy

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("agent0")
class Agent0(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Agent0, self).__init__()
        self.name = "Agent0"

        self.corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        self.x_squares = [(1, 1), (1, -2), (-2, 1), (-2, -2)]
        self.c_squares = [(0, 1), (0, -2), (1, 0), (1, -1), (-2, 0), (-2, -1), (-1, 1), (-1, -2)]

    def step(self, board, player, opponent):
        """
        Parameters
        ----------
        board : numpy.ndarray of shape (board_size, board_size)
            The board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).

        Returns
        -------
        move_pos : tuple of int
            The position (x, y) where the player places the disc.
        """

        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()

        best_score = float('-inf')
        best_move = None

        valid_moves = get_valid_moves(board, player)

        for move in valid_moves:
            simulated_board = copy.deepcopy(board)
            score = 0

            # n_flipped = count_capture(simulated_board, move, player)
            # score -= n_flipped

            execute_move(simulated_board, move, player)

            score += self.evaluate(simulated_board, player, opponent)

            if score > best_score:
                best_score = score
                best_move = move

        time_taken = time.time() - start_time

        return best_move

    def evaluate(self, board, player, opponent):
        score = 0

        for square in self.x_squares:
            disc = board[square]
            if disc == player:
                score -= 50
            elif disc == opponent:
                score += 50

        for square in self.c_squares:
            disc = board[square]
            if disc == player:
                score -= 30
            elif disc == opponent:
                score += 30

        for square in self.corners:
            disc = board[square]
            if disc == player:
                score += 100
            elif disc == opponent:
                score -= 100

        score -= len(get_valid_moves(board, opponent)) * 10

        return score

