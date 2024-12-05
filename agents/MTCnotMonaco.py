# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("MTCnotMonaco")
class MTC_NOT_MONACO(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MTC_NOT_MONACO, self).__init__()
        self.name = "MTC_NOT_MONACO"

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
        time_limit = 0.5
        move = self.monte_carlo_move(chess_board, player, opponent, time_limit, start_time)
        #print(f"Move is {move} in {time.time() - start_time} seconds.")
        return move

    class MoveNode:
        def __init__(self, move, state=None, parent=None):
            self.move = move
            self.state = state
            self.parent = parent
            self.visits = 0
            self.total_score = 0.0
            self.children = []

        def uct(self, total_parent_visits, exploration=1.414):
            if self.visits == 0:
                return float('inf')  # Always explore unvisited nodes
            return (self.total_score / self.visits) + exploration * (total_parent_visits ** 0.5 / (1 + self.visits))

        def is_fully_expanded(self):
            return len(self.children) > 0


    def monte_carlo_move(self, chess_board, player, opponent, time_limit, start_time):
        """
                Uses Monte Carlo simulations to determine the best move.
                """

        root = self.MoveNode(None, state=deepcopy(chess_board))

        while time.time() - start_time < time_limit:
            selected_node = self.select(root)

            if not selected_node.is_fully_expanded():
                self.expand(selected_node, player if selected_node.parent is None else opponent)

            # If no children were added (e.g., no valid moves), continue to next iteration
            if not selected_node.children:
                continue

            simulation_node = selected_node.children[np.random.randint(len(selected_node.children))]
            result = self.simulate_random_game(simulation_node.state, player, opponent)

            self.backpropagate(simulation_node, result)

        # Select the best move from the root's children
        if root.children:
            best_child = max(root.children, key=lambda child: child.visits)
            return best_child.move
        else:
            return None  # If no valid moves exist

    def select(self, node):
        """
        Traverse the tree to select a node using UCT.
        """
        while node.is_fully_expanded() and node.children:
            total_visits = sum(child.visits for child in node.children)
            node = max(node.children, key=lambda child: child.uct(total_visits))
        return node

    def expand(self, node, player):
        """
        Add all valid moves as children of the given node.
        If no valid moves, add a 'pass' child with the same state.
        """
        valid_moves = get_valid_moves(node.state, player)
        if valid_moves:
            for move in valid_moves:
                board_copy = deepcopy(node.state)
                execute_move(board_copy, move, player)
                node.children.append(self.MoveNode(move, state=board_copy, parent=node))
        else:
            # Add a 'pass' child to simulate passing the turn
            node.children.append(self.MoveNode(None, state=deepcopy(node.state), parent=node))

    def simulate_random_game(self, board, player, opponent):
        """
        Simulates a random game from the current board position.
        """

        # print("simulating random game")

        current_player = opponent  # Start with the opponent
        while not check_endgame(board, player, opponent)[0]:
            valid_moves = get_valid_moves(board, current_player)
            if valid_moves:
                # Heuristic: Prefer moves that maximize flips
                best_move = max(valid_moves, key=lambda move: count_capture(board, move, current_player))
                execute_move(board, best_move, current_player)
            current_player = player if current_player == opponent else opponent

        # Evaluate the final board state
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        return player_score - opponent_score

    def backpropagate(self, node, score):
        """
        Propagate the simulation result back up the tree.
        """
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

















    def evaluate_board(self, board, player, opponent):
        """
        Update the board with scores depending on the move that was executed
        """
        score = 0
        board_length = len(board[0]) - 1

        # Corners
        corners = [(0, 0), (0, board_length), (board_length, 0), (board_length, board_length)]

        # Attribute scores for the positions

        # check if we moved to a corner
        # Check if corners are empty or accessible before penalizing adjacency
        for r, c in corners:
            if board[r, c] == player:
                score += 100000

            if board[r, c] == opponent:  # elif or if??
                score -= 100000

            if board[r, c] == 0:  # Empty corner, adjacency is risky        # elif or if??
                # check if we are on a tile adjacent to a corner
                # adjacent to top left -> hardcode instead of r,c?
                if (r, c) == (0, 0):
                    if board[r + 1, c] == player or board[r + 1, c + 1] == player or board[r, c + 1] == player:  # x squares are the worst
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
        # captured_pieces = count_capture(board, move, player)
        # score += captured_pieces*50 # or 1000

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
        score -= opponent_best_score * 500  # Penalize if the opponent gains a strong position

        return score
