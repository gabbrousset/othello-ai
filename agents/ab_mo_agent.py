from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("ab_mo_agent")
class AB_MO_Agent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AB_MO_Agent, self).__init__()
        self.name = "AB_MO_Agent"

        self.start_time = None
        self.time_limit = .3

        self.transposition_table = {}

        self.corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        self.x_squares = [(1, 1), (1, -2), (-2, 1), (-2, -2)]
        self.c_squares = [(0, 1), (0, -2), (1, 0), (1, -1), (-2, 0), (-2, -1), (-1, 1), (-1, -2)]

    def has_time_left(self):
        return time.time() - self.start_time < self.time_limit

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
        self.start_time = time.time()

        best_move = None

        depth = 1
        while self.has_time_left():
            try:
                best_move = self.ids(board, player, opponent, depth)
                depth += 1
            except TimeoutError:
                break

        # time_taken = time.time() - self.start_time
        # print('ab_mo time taken:', time_taken, 'at depth:', depth)

        return best_move

    def get_ordered_moves(self, board, player, opponent):
        moves = get_valid_moves(board, player)
        # using enum to avoid comparing moves (i think its faster)
        move_scores = [(self.score_move(board, move, player, opponent), i, move) for i, move in enumerate(moves)]
        move_scores.sort(reverse=True)
        return [move for score, i, move in move_scores]

    def score_move(self, board, move, player, opponent):
        new_board = deepcopy(board)
        execute_move(new_board, move, player)
        return self.evaluate(new_board, player, opponent)

    def ids(self, board, player, opponent, depth):
        moves = self.get_ordered_moves(board, player, opponent)
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta(new_board, depth - 1, -beta, -alpha, opponent, player)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def alpha_beta(self, board, depth, alpha, beta, player, opponent):
        """
        Returns an eval_score
        """
        if not self.has_time_left():
            raise TimeoutError

        is_endgame, p1_score, p2_score = check_endgame(board, player, opponent)

        if is_endgame:
            return self.evaluate_endgame(p1_score, p2_score, player)

        if depth == 0:
            return self.evaluate(board, player, opponent)

        moves = self.get_ordered_moves(board, player, opponent)

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
            return (p1_score - p2_score) * 1000
        return (p2_score - p1_score) * 1000

