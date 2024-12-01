from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("ab_agent")
class AB_Agent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AB_Agent, self).__init__()
        self.name = "AB_Agent"
        self.time_limit = 1.9

        self.transposition_table = {}

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

        moves = get_valid_moves(board, player)
        best_move = None

        depth = 3
        alpha = float('-inf')
        beta = float('inf')

        best_value = float('-inf')

        for move in moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta(new_board, depth - 1, -beta, -alpha, opponent, player)

            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, value)

        time_taken = time.time() - start_time

        # Dummy return (you should replace this with your actual logic)
        # Returning a random valid move as an example
        return best_move

    def alpha_beta(self, board, depth, alpha, beta, player, opponent):
        """
        Returns an eval_score
        """
        is_endgame, p1_score, p2_score = check_endgame(board, player, opponent)

        if is_endgame:
            return self.evaluate_endgame(p1_score, p2_score, player)

        if depth == 0:
            return self.evaluate(board, player, opponent)

        moves = get_valid_moves(board, player)

        if not moves:
            return -self.alpha_beta(board, depth - 1, -beta, -alpha, opponent, player)

        for move in moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta(new_board, depth - 1, -beta, -alpha, opponent, player)

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return alpha

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

    def evaluate_endgame(self, p1_score, p2_score, player):
        if player == 1:
            return p1_score - p2_score
        return p2_score - p1_score

