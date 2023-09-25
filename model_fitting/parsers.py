from functools import total_ordering
from fourbynine import fourbynine_board, fourbynine_pattern, fourbynine_move, Player_Player1, Player_Player2, bool_to_player, player_to_string
from model_fit import bads_parameters_to_model_parameters
import json


@total_ordering
class CSVMove:
    """
    Encodes a single move made by a player at a given position.
    """
    @staticmethod
    def create(line):
        """
        Creates a CSVMove from a single CSV line. The format of the CSV string should be:

        black_pieces white_pieces player move time [group_id] participant_id, where group_id is optional.

        black_pieces and white_pieces should both be base-10 numbers encoding the positions of all of the black and white pieces on the board respectively as a bitfield, with the LSB corresponding to the upper left corner of the board, i.e. 0b1 corresponding to a piece at index 0.
        player should be either the string Black, White, black, white, 0, or 1, where 0 corresponds to black and 1 corresponds to white.
        move should be a base-10 number encoding the position at which the move being specified is played, i.e. 1 corresponding to a move at index 0, 2 to a move at index 1, 4 to index 2, etc.
        time is the time in milliseconds that the player took to play the move.
        group_id is an optional integer encoding a group ID for the player.
        participant_id is a string denoting the given player.

        Args:
            line: The line to parse.

        Returns: A CSVMove corresponding to the given line.
        """
        # Try splitting by comma
        parameters = line.rstrip().split(',')
        if (len(parameters) == 1):
            # Otherwise, try spltting by space.
            parameters = line.rstrip().split()
        if (len(parameters) >= 6):
            board = fourbynine_board(fourbynine_pattern(
                int(parameters[0])), fourbynine_pattern(int(parameters[1])))

            def parse_player(player):
                if player.lower() == "white" or player == '1':
                    return True
                if player.lower() == "black" or player == '0':
                    return False
                raise Exception("Unrecognized player token: {}".format(player))

            player = parse_player(parameters[2])
            if (player != board.active_player()):
                raise ValueError("Given player {} is not the active player on the given board: {}".format(
                    player_to_string(player), board.to_string()))

            def move_bitfield_to_index(move):
                """
                Takes a bitfield representing a move and converts it to its corresponding tile index.

                Args:
                    move: A move encoded as a one-hot bitfield with the single 1 corresponding to the moves position on the board.

                Returns:
                    The index of the move, where index 0 corresponds to the upper left of the board, incrementing in a row-major fashion.
                """
                if int(move).bit_count() != 1:
                    raise Exception(
                        "Invalid move given: {}. Moves are expected to be in bitfield format with a single bit set!".format(move))
                return int(move).bit_length() - 1

            move = fourbynine_move(move_bitfield_to_index(
                parameters[3]), 0.0, board.active_player())
            time = float(parameters[4])
            if (len(parameters) == 6):
                group_id = 1
                participant_id = parameters[5]
            else:
                group_id = int(parameters[5])
                participant_id = parameters[6]
            return CSVMove(board, move, time, group_id, participant_id)
        else:
            raise Exception(
                "Given input has incorrect number of parameters (expected 6 or 7): " + csv_string)

    def __init__(
            self,
            board,
            move,
            time,
            group_id,
            participant_id):
        """
        Construct a move.

        Args:
            board: The board that the move was played on as a fourbynine_board object.
            move: The move that was made as a fourbynine_move object.
            time: The amount of time it took to play this move in milliseconds.
            group_id: The integer group that this player belonged to.
            participant_id: A string identifying the player.
        """
        self.board = board
        self.move = move
        # Test that the move is valid. This will throw if it isn't.
        _ = self.board + self.move
        self.player = board.active_player()
        self.time = float(time)
        self.group_id = int(group_id)
        self.participant_id = str(participant_id)

    def __repr__(self):
        """
        Returns:
            A valid CSV string representing the given move.
        """
        return "\t".join([str(int(self.board.get_pieces(Player_Player1).to_string(), 2)), str(int(self.board.get_pieces(Player_Player2).to_string(), 2)), player_to_string(self.player), str(2**self.move.board_position), str(self.time), str(self.group_id), self.participant_id])

    def __hash__(self):
        """
        Hashes the move.
        """
        return hash(str(self))

    def __eq__(self, other):
        """
        Returns:
            True if both moves are equivalent.
        """
        return str(self) == str(other)

    def __lt__(self, other):
        """
        Returns:
            True if a given move is "less than" another move. Ordering is mostly arbitrary, this function just establishes a canonical ordering.
        """
        return str(self) < str(other)

    def __getstate__(self):
        return str(self)

    def __setstate__(self, state_string):
        new_state = CSVMove.create(state_string)
        self.__dict__ = new_state.__dict__


def _parse_participant_csv(lines, group_id=1):
    """
    Parses a list of CSV-encoded moves.

    Args:
        lines: A list of CSV strings encoding moves.
        group_id: An optional group_id, used if the CSV line does not specify one.

    Returns:
        A list of CSVMove objects, one for each valid line passed in.
    """
    return [CSVMove.create(line) for line in lines]


def _parse_participant_json(json_file, group_id=1, participant_id="1"):
    """
    Parses a JSON string encoding games.

    Args:
        json_file: The file containing moves. The moves are expected to be at root["free_play"]["solution"] encoded as a string in the format "<first_move_index>-<second_move_index>-<third_move_index>-...".
                   The color that the player was playing should be at root["free_play"]["player_color"], with the times each move took at root["free_play"]["all_move_RT"].
        group_id: The group ID to assign to all games in this file.
        participant_id: The name of the participant for all games in this file.

    Returns:
        A list of CSVMove objects, one for each move encoded in the JSON file.
    """
    root = json.loads(json_file)
    moves = []
    for game in root["free_play"]:
        if not game:
            continue
        board = fourbynine_board()
        solution = game["solution"].split("-")
        input_player = game["player_color"].lower() == "white"
        times = game["all_move_RT"]
        player = False  # Corresponds to Black
        game_valid = True
        candidate_moves = []
        for p in solution:
            move = fourbynine_move(int(p), 0.0, bool_to_player(player))
            try:
                if player == input_player:
                    candidate_moves.append(
                        CSVMove(board, move, times[len(candidate_moves)], group_id, participant_id))
                board += move
            except Exception as exce:
                print("Skipping solution {} as it is malformed.".format(solution))
                game_valid = False
                break
            player = not player
        if game_valid:
            moves.extend(candidate_moves)
    return moves


def parse_participant_file(f, group_id=1, participant_id="1"):
    """
    Parses a file by first attempting to parse it as a JSON file, and then falling back to a CSV file.

    Args:
        f: The path to the file to parse.
        group_id: The group ID to assign to all games in this file. If the file is a CSV file, ignored if the CSV contains group information.
        participant_id: The name of the participant for all games in this file. Ignored if the file is a CSV file.
    """
    with open(f, 'r') as lines:
        try:
            return _parse_participant_json(lines.read(), group_id, participant_id)
        except json.JSONDecodeError:
            print(
                "File is either not a JSON file, or is malformed. Attempting to parse as a CSV...")
            lines.seek(0)
        return _parse_participant_csv(lines, group_id)


def parse_bads_parameter_file_to_model_parameters(f):
    """
    Parses a file containing a list of comma-separated parameters for a model into a Python list.

    Args:
        f: The path to the file to parse.
    """
    with open(f, 'r') as lines:
        for line in lines:
            if line.startswith("#"):
                continue
            return bads_parameters_to_model_parameters(line.split(","))
