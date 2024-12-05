from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import count_capture, execute_move, check_endgame, get_valid_moves

# TODO remove import before submitting to contest
import os
import psutil


@register_agent("beaver")
class Beaver(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Beaver, self).__init__()
        self.name = "Beaver"

        self.start_time = None
        self.time_limit = .3

        self.EXACT = 0
        self.LOWERBOUND = 1 # position is at least this good
        self.UPPERBOUND = 2 # position is at most this bad
        self.transposition_table = {}

        self.MAX_MEMORY_MB = 400
        # self.process = psutil.Process(os.getpid())
        self.table_size_limit = 1000000
        self.move_number = 0
        self.max_table_v_age = 4    # only keep positions from last 2 moves

        self.first_run = True
        self.M = None
        # self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        self.corners = []
        self.x_squares = []
        self.c_squares = []
        self.edges_close = []
        self.center_corners = []
        
        self.board_weights = None
        
        self.weights = {
            'normal': -5,
            'corner': 100,
            'x_square': -50,
            'c_square': -30,
            'edge': 10,
            'edge_close': 20,
            'inner_center': 5,
            'center': 3,
            'center_corner': 15,

            # 'stability': 100,
            # 'frontier': -15,
            # 'mobility': 20,
            # 'disc_diff': 5,
        }

        # self.stage_multipliers = {
        # }

        self.num_of_nodes_checked = 0
        self.table_hits = 0

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
        self.num_of_nodes_checked = 0
        self.table_hits = 0

        self.move_number += 1
        if self.move_number > self.max_table_v_age:
            self.transposition_table.clear()

        if self.first_run:
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

            self.first_run = False

        best_move = None

        depth = 1
        while self.has_time_left():
            try:
                best_move = self.ids(board, player, opponent, depth)
                depth += 1
            except TimeoutError:
                break

        time_taken = time.time() - self.start_time
        print('beaver time taken:', time_taken, 'at depth:', depth, 'tt:', len(self.transposition_table), 'hits', self.table_hits, 'nodes:', self.num_of_nodes_checked)
        process = psutil.Process()
        mem_info = process.memory_info()
        # print(f'memory usage: {mem_info.rss / (1024*1024):.2f} MB')

        return best_move
    
    def initialize_weights(self, dimensions):
        # set edges
        self.board_weights = np.full(dimensions, self.weights['edge'])
        # set inner values of rows, columns
        self.board_weights[1:-1, 1:-1] = self.weights['normal']

        # 6x6 = [2, 4] =   [2, M-2]
        # 8x8 = [2, 6] =   [2, M-2]
        # 10x10 = [4, 6] = [4, M-4]
        # 12x12 = [4, 8] = [4, M-4]
        self.board_weights[2:self.M - 2, 2:self.M - 2] = self.weights['center']
        if self.M > 7:
            for center_corner in self.center_corners:
                self.board_weights[center_corner] = self.weights['center_corner']

            if self.M > 9:
                self.board_weights[4:self.M-4, 4:self.M-4] = self.weights['inner_center']

        for corner in self.corners:
            self.board_weights[corner] = self.weights['corner']

        for x_square in self.x_squares:
            self.board_weights[x_square] = self.weights['x_square']

        for c_square in self.c_squares:
            self.board_weights[c_square] = self.weights['c_square']

        if self.M > 6:
            for edge_close in self.edges_close:
                self.board_weights[edge_close] = self.weights['edge_close']

    def get_ordered_moves(self, board, player):
        moves = get_valid_moves(board, player)
        # using enum to avoid comparing moves (i think its faster)
        move_scores = [(self.score_move(board, move, player), i, move) for i, move in enumerate(moves)]
        move_scores.sort(reverse=True)
        return [move for score, i, move in move_scores]

    def score_move(self, board, move, player):
        score = self.board_weights[move]
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

    def alpha_beta_negamax(self, board, depth, alpha, beta, player, opponent, num_moves=0):
        """
        Returns an eval_score
        """
        if not self.has_time_left():
            raise TimeoutError

        alpha_og = alpha

        board_hash = self.get_board_hash(board, player)
        tt_entry = self.transposition_table.get(board_hash)

        if tt_entry and tt_entry['d'] >= depth:
            self.table_hits += 1
            if tt_entry['f'] == self.EXACT:
                return tt_entry['v']
            if tt_entry['f'] == self.LOWERBOUND:
                alpha = max(alpha, tt_entry['v'])
            elif tt_entry['f'] == self.UPPERBOUND:
                beta = min(beta, tt_entry['v'])

            if alpha >= beta:
                return tt_entry['v']

        is_endgame, p1_score, p2_score = check_endgame(board, player, opponent)

        flag = self.EXACT

        if is_endgame:
            value = self.evaluate_endgame(p1_score, p2_score, player)
            self.store_position(board_hash, depth, value, flag)
            return value

        if depth == 0:
            value = self.evaluate(board, player, opponent, num_moves)
            self.store_position(board_hash, depth, value, flag)
            return value

        moves = self.get_ordered_moves(board, player)

        if not moves:
            value = -self.alpha_beta_negamax(board, depth - 1, -beta, -alpha, opponent, player)
            self.store_position(board_hash, depth, value, flag)
            return value

        new_board = np.empty_like(board)

        for move in moves:
            np.copyto(new_board, board)
            execute_move(new_board, move, player)

            value = -self.alpha_beta_negamax(new_board, depth - 1, -beta, -alpha, opponent, player, num_moves=len(moves))
            alpha = max(alpha, value)

            if alpha >= beta:
                # position is too good
                flag = self.LOWERBOUND
                self.store_position(board_hash, depth, alpha, flag)
                return alpha

        if alpha <= alpha_og:
            flag = self.UPPERBOUND

        self.store_position(board_hash, depth, alpha, flag)
        return alpha

    def evaluate(self, board, player, opponent, num_moves):
        self.num_of_nodes_checked += 1
        score = 0

        player_pattern = np.where(board == player, 1, 0)
        opp_pattern = np.where(board == opponent, 1, 0)

        score += (self.board_weights * player_pattern).sum() - (self.board_weights * opp_pattern).sum()

        score += (num_moves - len(get_valid_moves(board, opponent))) * 10

        return score

    @staticmethod
    def evaluate_endgame(p1_score, p2_score, player):
        # not returning infinite values because i get errors when comparing in negamax
        if player == 1:
            return (p1_score - p2_score) << 17  # shifting bc i think its faster than multiplying
        return (p2_score - p1_score) << 17

    @staticmethod
    def get_board_hash(board, player):
        return hash((board.tobytes(), player))

    def store_position(self, board_hash, depth, value, flag):
        if len(self.transposition_table) >= self.table_size_limit:
            # removing 10% of tables, from oldest to newest entries
            remove_count = self.table_size_limit // 10
            for _ in range(remove_count):
                self.transposition_table.pop(next(iter(self.transposition_table)))

        self.transposition_table[board_hash] = {
            'd': depth,
            'v': value,
            'f': flag,
        }
