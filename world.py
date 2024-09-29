import numpy as np
from copy import deepcopy
import traceback
from agents import *
from ui import UIEngine
from time import sleep, time
import click
import logging
from store import AGENT_REGISTRY
from constants import *
import sys

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


class World:
    def __init__(
        self,
        player_1="random_agent",
        player_2="random_agent",
        board_size=None,
        display_ui=False,
        display_delay=2,
        display_save=False,
        display_save_path=None,
        autoplay=False,
    ):
        """
        Initialize the game world

        Parameters
        ----------
        player_1: str
            The registered class of the first player
        player_2: str
            The registered class of the second player
        board_size: int
            The size of the board. If None, board_size = a number between MIN_BOARD_SIZE and MAX_BOARD_SIZE
        display_ui : bool
            Whether to display the game board
        display_delay : float
            Delay between each step
        display_save : bool
            Whether to save an image of the game board
        display_save_path : str
            The path to save the image
        autoplay : bool
            Whether the game is played in autoplay mode
        """
        # Two players
        logger.info("Initialize the game world")
        # Load agents as defined in decorators
        self.player_1_name = player_1
        self.player_2_name = player_2
        if player_1 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_1}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )
        if player_2 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_2}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )

        p0_agent = AGENT_REGISTRY[player_1]
        p1_agent = AGENT_REGISTRY[player_2]
        logger.info(f"Registering p0 agent : {player_1}")
        self.p0 = p0_agent()
        logger.info(f"Registering p1 agent : {player_2}")
        self.p1 = p1_agent()

        # check autoplay
        if autoplay:
            if not self.p0.autoplay or not self.p1.autoplay:
                raise ValueError(
                    f"Autoplay mode is not supported by one of the agents ({self.p0} -> {self.p0.autoplay}, {self.p1} -> {self.p1.autoplay}). Please set autoplay=True in the agent class."
                )

        self.player_names = {PLAYER_1_ID: PLAYER_1_NAME, PLAYER_2_ID: PLAYER_2_NAME}

        if board_size is None:
            # Random board size, ensuring even numbers for Reversi Othello
            self.board_size = np.random.choice([6, 8, 10, 12])
            logger.info(
                f"No board size specified. Randomly generating size: {self.board_size}x{self.board_size}"
            )
        else:
            self.board_size = board_size
            logger.info(f"Setting board size to {self.board_size}x{self.board_size}")

        # Initialize the game board (0: empty, 1: black disc, 2: white disc)
        self.chess_board = np.zeros((self.board_size, self.board_size), dtype=int)

        # Initialize the center discs for Reversi Othello
        mid = self.board_size // 2
        self.chess_board[mid - 1][mid - 1] = 2  # White
        self.chess_board[mid - 1][mid] = 1      # Black
        self.chess_board[mid][mid - 1] = 1      # Black
        self.chess_board[mid][mid] = 2          # White

        # Whose turn to step
        self.turn = 0

        # Time taken by each player
        self.p0_time = []
        self.p1_time = []

        # Cache to store and use the data
        self.results_cache = ()
        # UI Engine
        self.display_ui = display_ui
        self.display_delay = display_delay
        self.display_save = display_save
        self.display_save_path = display_save_path

        if display_ui:
            # Initialize UI Engine
            logger.info(
                f"Initializing the UI Engine, with display_delay={display_delay} seconds"
            )
            self.ui_engine = UIEngine(self.board_size, self)
            self.render()

    def get_current_player(self):
        """
        Get the current player (1: Black, 2: White)
        """
        return 1 if self.turn == 0 else 2

    def update_player_time(self, time_taken):
        """
        Update the time taken by the player

        Parameters
        ----------
        time_taken : float
            Time taken by the player
        """
        if not self.turn:
            self.p0_time.append(time_taken)
        else:
            self.p1_time.append(time_taken)

    def step(self):
        """
        Take a step in the game world.
        Runs the agents' step function and updates the game board accordingly.
        If the agents' step function raises an exception, the step will be replaced by a Random Move.

        Returns
        -------
        results: tuple
            The results of the step containing (is_endgame, player_1_score, player_2_score)
        """
        cur_player = self.get_current_player()
        opponent = 3 - cur_player  # 1 if current player is 2, 2 if current player is 1

        try:
            # Run the agent's step function
            start_time = time()
            move_pos = self.get_current_agent().step(
                deepcopy(self.chess_board),
                cur_player,
                opponent,
            )
            time_taken = time() - start_time
            self.update_player_time(time_taken)

            if not self.is_valid_move(move_pos, cur_player):
                raise ValueError(f"Invalid move by player {cur_player}: {move_pos}")

        except BaseException as e:
            ex_type = type(e).__name__
            if (
                "SystemExit" in ex_type and isinstance(self.get_current_agent(), HumanAgent)
            ) or "KeyboardInterrupt" in ex_type:
                sys.exit(0)
            print(
                "An exception raised. The traceback is as follows:\n{}".format(
                    traceback.format_exc()
                )
            )
            print("Executing Random Move!")
            move_pos = self.random_move(cur_player)

            ########
            if move_pos is None:
                print(f"Player {cur_player} has no valid moves. Ending the game.")

                p0_score = np.sum(self.chess_board == 1)
                p1_score = np.sum(self.chess_board == 2)
    
                results = True, p0_score, p1_score
                self.results_cache = results

                # Render board and show results
                if self.display_ui:
                    self.render()
                    if results[0]:
                        click.echo("Press a button to exit the game.")
                        try:
                            _ = click.getchar()
                        except:
                            _ = input()

                return results
        

            ############

        # Execute move
        self.execute_move(move_pos, cur_player)
        logger.info(
            f"Player {self.player_names[self.turn]} places at {move_pos}. Time taken this turn (in seconds): {time_taken}"
        )

        # Change turn
        self.turn = 1 - self.turn

        results = self.check_endgame()
        self.results_cache = results

        # Render board and show results
        if self.display_ui:
            self.render()
            if results[0]:
                click.echo("Press a button to exit the game.")
                try:
                    _ = click.getchar()
                except:
                    _ = input()

        return results

    def is_valid_move(self, move_pos, player):
        """
        Check if the move is valid (i.e., it captures at least one opponent's disc).

        Parameters
        ----------
        move_pos : tuple
            The position where the player wants to place a disc
        player : int
            The current player (1: Black, 2: White)

        Returns
        -------
        bool
            Whether the move is valid
        """
        r, c = move_pos
        if self.chess_board[r, c] != 0:
            return False

        # Check if move captures any opponent discs in any direction
        for move in self.get_directions():
            if self.check_capture(move_pos, player, move):
                return True

        return False

    def check_capture(self, move_pos, player, direction):
        """
        Check if placing a disc at move_pos captures any discs in the specified direction.

        Parameters
        ----------
        move_pos : tuple
            The position where the player places the disc
        player : int
            The current player (1: Black, 2: White)
        direction : tuple
            The direction to check (dx, dy)

        Returns
        -------
        bool
            Whether discs can be captured in the specified direction
        """
        r, c = move_pos
        dx, dy = direction
        r += dx
        c += dy
        captured = []

        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.chess_board[r, c] == 0:
                return False
            if self.chess_board[r, c] == player:
                return len(captured) > 0
            captured.append((r, c))
            r += dx
            c += dy

        return False

    def execute_move(self, move_pos, player):
        """
        Execute the move and flip the opponent's discs accordingly.

        Parameters
        ----------
        move_pos : tuple
            The position where the player places the disc
        player : int
            The current player (1: Black, 2: White)
        """
        r, c = move_pos
        self.chess_board[r, c] = player

        # Flip opponent's discs in all directions where captures occur
        for direction in self.get_directions():
            if self.check_capture(move_pos, player, direction):
                self.flip_discs(move_pos, player, direction)

    def flip_discs(self, move_pos, player, direction):
        """
        Flip the discs in the specified direction.

        Parameters
        ----------
        move_pos : tuple
            The position where the player places the disc
        player : int
            The current player (1: Black, 2: White)
        direction : tuple
            The direction to check (dx, dy)
        """
        r, c = move_pos
        dx, dy = direction
        r += dx
        c += dy

        while self.chess_board[r, c] != player:
            self.chess_board[r, c] = player
            r += dx
            c += dy

    def check_endgame(self):
        """
        Check if the game ends and compute the final score.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        if not np.any(self.chess_board == 0):  # No empty spaces left
            p0_score = np.sum(self.chess_board == 1)
            p1_score = np.sum(self.chess_board == 2)
            return True, p0_score, p1_score

        return False, np.sum(self.chess_board == 1), np.sum(self.chess_board == 2)

    def get_directions(self):
        """
        Get all directions (8 directions: up, down, left, right, and diagonals)

        Returns
        -------
        list of tuple
            List of direction vectors
        """
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def random_move(self, player):
        """
        Randomly select a valid move.

        Parameters
        ----------
        player : int
            The current player (1: Black, 2: White)

        Returns
        -------
        tuple
            The position to place the disc
        """
        valid_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.is_valid_move((r, c), player):
                    valid_moves.append((r, c))
        
        """
        """
        #### 
        if len(valid_moves) == 0:
            # If no valid moves are available, return None
            print(f"No valid moves left for player {player}.")
            return None
        

        
        
        return valid_moves[np.random.randint(0, len(valid_moves))]

    def get_current_agent(self):
        """
        Get the current player's agent

        Returns
        -------
        agent : object
            The agent object of the current player
        """
        return self.p0 if self.turn == 0 else self.p1

    def render(self, debug=False):
        """
        Render the game board using the UI Engine
        """
        self.ui_engine.render(self.chess_board, debug=debug)
        sleep(self.display_delay)


if __name__ == "__main__":
    world = World()
    is_end, p0_score, p1_score = world.step()
    while not is_end:
        is_end, p0_score, p1_score = world.step()
    print(p0_score, p1_score)
