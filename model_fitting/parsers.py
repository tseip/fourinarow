from functools import total_ordering
from fourbynine import fourbynine_board, fourbynine_move, Player_Player1, Player_Player2, bool_to_player, player_to_string
import json


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


@total_ordering
class CSVMove:
    """
    Encodes a single move made by a player at a given position.
    """

    def __init__(
            self,
            black_pieces,
            white_pieces,
            player,
            move,
            time,
            group,
            participant):
        """
        Construct a move.

        Args:
            black_pieces: A base-10 number encoding the positions of all of the black pieces in the position the given move was played.
            white_pieces: A base-10 number encoding the positions of all of the white pieces in the position the given move was played.
            player: Which player made the move at the given position - True corresponds to White, False to Black.
            move: The move that was made. This move is expected to be encoded as an index into the board representing where the move was played, with the top left corner indexed at 0 and incrementing row-wise.
            time: The amount of time it took to play this move.
            group: The integer group that this player belonged to.
            participant: A string identifying the player.
        """
        self.black_pieces = int(black_pieces)
        self.white_pieces = int(white_pieces)
        self.player = bool(player)
        self.move = int(move)
        self.time = float(time)
        self.group = int(group)
        self.participant = str(participant)

    def __hash__(self):
        """
        Hashes the move.
        """
        return hash(
            (self.black_pieces,
             self.white_pieces,
             self.player,
             self.move,
             self.time,
             self.participant))

    def __repr__(self):
        """
        Returns:
            A valid CSV string representing the given move.
        """
        return "\t".join([str(self.black_pieces), str(self.white_pieces), player_to_string(self.player), str(2**self.move), str(self.time), str(self.group), self.participant])

    def __eq__(self, other):
        """
        Returns:
            True if both moves are equivalent.
        """
        return self.black_pieces == other.black_pieces and self.white_pieces == other.white_pieces and self.player == other.player and self.move == other.move and self.time == other.time and self.group == other.group and self.participant == other.participant

    def __lt__(self, other):
        """
        Returns:
            True if a given move is "less than" another move. Ordering is mostly arbitrary, this function just establishes a canonical ordering.
        """
        return (self.black_pieces, self.white_pieces, self.player, self.move) < (other.black_pieces, other.white_pieces, other.player, other.move)


def _parse_participant_csv(lines, group_id=1):
    """
    Parses a list of CSV-encoded moves.

    Args:
        lines: A list of CSV strings encoding moves.

    Returns:
        A list of CSVMove objects, one for each valid line passed in.
    """
    def parse_player(player):
        if player.lower() == "white" or player == '1':
            return True
        if player.lower() == "black" or player == '0':
            return False
        raise Exception("Unrecognized player token: {}".format(player))

    moves = []
    for line in lines:
        # Try splitting by comma
        parameters = line.rstrip().split(',')
        if (len(parameters) == 1):
            parameters = line.rstrip().split()
        if (len(parameters) == 6):
            moves.append(CSVMove(
                parameters[0], parameters[1], parse_player(parameters[2]), move_bitfield_to_index(parameters[3]), parameters[4], group_id, parameters[5]))
        elif (len(parameters) == 7):
            moves.append(CSVMove(parameters[0], parameters[1], parse_player(parameters[2]),
                                 move_bitfield_to_index(parameters[3]), parameters[4], parameters[5], parameters[6]))
        else:
            raise Exception(
                "Given input has incorrect number of parameters (expected 6 or 7): " + line)
    return moves


def _parse_participant_json(json_file, group_id=1, participant="1"):
    """
    Parses a JSON string encoding games.

    Args:
        json_file: The file containing moves. The moves are expected to be at root["free_play"]["solution"] encoded as a string in the format "<first_move_index>-<second_move_index>-<third_move_index>-...".
                   The color that the player was playing should be at root["free_play"]["player_color"], with the times each move took at root["free_play"]["all_move_RT"].
        group_id: The group ID to assign to all games in this file.
        participant: The name of the participant for all games in this file.

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
            if player == input_player:
                candidate_moves.append(CSVMove(int(board.get_pieces(Player_Player1).to_string(), 2), int(board.get_pieces(
                    Player_Player2).to_string(), 2), bool_to_player(input_player), int(p), times[len(candidate_moves)], group_id, participant))
            try:
                board += fourbynine_move(int(p), 0.0, bool_to_player(player))
            except Exception:
                print("Skipping solution: {}".format(solution))
                game_valid = False
                break
            player = not player
        if game_valid:
            moves.extend(candidate_moves)
    return moves


def parse_participant_file(f, group_id=1, participant="1"):
    """
    Parses a file by first attempting to parse it as a JSON file, and then falling back to a CSV file.

    Args:
        f: The path to the file to parse.
        group_id: The group ID to assign to all games in this file. If the file is a CSV file, ignored if the CSV contains group information.
        participant: The name of the participant for all games in this file. Ignored if the file is a CSV file.
    """
    with open(f, 'r') as lines:
        try:
            return _parse_participant_json(lines.read(), group_id, participant)
        except json.JSONDecodeError:
            print(
                "File is either not a JSON file, or is malformed. Attempting to parse as a CSV...")
            lines.seek(0)
        return _parse_participant_csv(lines, group_id)
