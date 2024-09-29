import numpy as np
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name
@register_agent("random_agent")
class RandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True

    def step(self, chess_board, player, opponent):
        """
        Randomly selects a valid position to place a disc.

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
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
        board_size = chess_board.shape[0]

        # Build a list of valid moves (empty spots on the board)
        valid_moves = []
        for r in range(board_size):
            for c in range(board_size):
                if chess_board[r, c] == 0:  # Valid move if the spot is empty
                    valid_moves.append((r, c))

        # Randomly select a move from the list of valid moves
        move_pos = valid_moves[np.random.randint(0, len(valid_moves))]

        return move_pos
