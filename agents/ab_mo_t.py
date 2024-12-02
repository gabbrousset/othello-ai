from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("ab_mo_t")
class AB_MO_T(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AB_MO_T, self).__init__()
        self.name = "AB_MO_T"

        self.start_time = None
        self.time_limit = .3

        self.transposition_table = {}

        self.first_run = True
        self.M = None
        
        self.corners = []
        self.x_squares = []
        self.c_squares = []
        self.edges_close = []
        self.center_corners = []
        
        self.weights = None
        self.normal_w = -5
        self.corner_w = 100
        self.x_square_w = -50
        self.c_square_w = -30
        self.edge_w = 5
        self.edge_close_w = 20
        self.inner_center_w = 5
        self.center_w = 3
        self.center_corner_w = 15

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

        if self.first_run:
            self.first_run = False
            
            dimensions = board.shape
            self.M = dimensions[0]
            last_idx = self.M - 1
            
            self.corners = {(0, 0), (0, last_idx), (last_idx, 0), (last_idx, last_idx)}
            self.x_squares = {(1, 1), (1, self.M - 2), (self.M - 2, 1), (self.M - 2, self.M - 2)}
            self.c_squares = {(0, 1), (0, self.M - 2), (1, 0), (1, last_idx), (self.M - 2, 0),
                              (self.M - 2, last_idx), (last_idx, 1), (last_idx, self.M - 2)}
            self.edges_close = {(0, 2), (0, self.M - 3), (2, 0), (2, last_idx), (self.M - 3, 0),
                              (self.M - 3, last_idx), (last_idx, 2), (last_idx, self.M - 3)}
            self.center_corners = {(2, 2), (2, self.M -3), (self.M - 3, 2), (self.M - 3, self.M - 3)}

            self.initialize_weights(dimensions)

        best_move = None

        depth = 1
        while self.has_time_left():
            try:
                best_move = self.ids(board, player, opponent, depth)
                depth += 1
            except TimeoutError:
                break

        time_taken = time.time() - self.start_time
        print('ab_mo_t time taken:', time_taken, 'at depth:', depth)

        return best_move
    
    def initialize_weights(self, dimensions):
        # set edges
        self.weights = np.full(dimensions, self.edge_w)
        # set inner values of rows, columns
        self.weights[1:-1, 1:-1] = self.normal_w

        # 6x6 = [2, 4] =   [2, M-2]
        # 8x8 = [2, 6] =   [2, M-2]
        # 10x10 = [4, 6] = [4, M-4]
        # 12x12 = [4, 8] = [4, M-4]
        self.weights[2:self.M - 2, 2:self.M - 2] = self.center_w
        if self.M > 7:
            for center_corner in self.center_corners:
                self.weights[center_corner] = self.center_corner_w

            if self.M > 9:
                self.weights[4:self.M-4, 4:self.M-4] = self.inner_center_w

        for corner in self.corners:
            self.weights[corner] = self.corner_w

        for x_square in self.x_squares:
            self.weights[x_square] = self.x_square_w

        for c_square in self.c_squares:
            self.weights[c_square] = self.c_square_w

        if self.M > 6:
            for edge_close in self.edges_close:
                self.weights[edge_close] = self.edge_close_w

    def get_ordered_moves(self, board, player):
        moves = get_valid_moves(board, player)
        # using enum to avoid comparing moves (i think its faster)
        move_scores = [(self.score_move(board, move, player), i, move) for i, move in enumerate(moves)]
        move_scores.sort(reverse=True)
        return [move for score, i, move in move_scores]

    def score_move(self, board, move, player):
        score = self.weights[move]
        # score += count_capture(board, move, player) * 3
        return score

    def ids(self, board, player, opponent, depth):
        moves = self.get_ordered_moves(board, player)
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        new_board = np.empty_like(board)

        for move in moves:
            np.copyto(new_board, board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta_negamax(new_board, depth - 1, -beta, -alpha, opponent, player)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def alpha_beta_negamax(self, board, depth, alpha, beta, player, opponent):
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

        moves = self.get_ordered_moves(board, player)

        if not moves:
            return -self.alpha_beta_negamax(board, depth - 1, -beta, -alpha, opponent, player)

        new_board = np.empty_like(board)

        for move in moves:
            np.copyto(new_board, board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta_negamax(new_board, depth - 1, -beta, -alpha, opponent, player)

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
        # not returning infinite values because i get errors when comparing in negamax
        if player == 1:
            return (p1_score - p2_score) << 17  # shifting bc i think its faster than multiplying
        return (p2_score - p1_score) << 17
