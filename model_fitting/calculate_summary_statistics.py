from fourbynine import *
import matplotlib.pyplot as plt
import numpy as np
from parsers import parse_participant_file, parse_bads_parameter_file_to_model_parameters
from board_explorer import pattern_string_to_board_positions
from feature_utilities import array_to_pattern
import argparse
import random
import fourbynine
from tqdm import tqdm


def index_to_coordinate(index):
    """
    Converts a move index to x, y coordinates.

    Args:
        index: The index of the move. 0 is the upper left corner.

    Returns:
        A (row, col) pair corresponding to the row, column of the index. (0, 0) is the upper left corner.
    """
    m = fourbynine.fourbynine_move(int(index), 0.0, fourbynine.Player_Player1)
    return np.array([m.get_row(), m.get_col()])


def manhattan_distance(coordinate_1, coordinate_2):
    """
    Calculates the Manhattan (L1) distance of two coordinates.

    Args:
        coordinate_1: The first coordinate.
        coordinate_2: The second coordinate.

    Returns:
        The L1 distance between the two coordinates, or 0.0 if one of the inputs is None.
    """
    if coordinate_1 is None or coordinate_2 is None:
        return 0.0
    return np.sum(np.abs(coordinate_1 - coordinate_2))


def average_manhattan_distance_from_pattern(pattern, move):
    """
    Calculates the average Manhattan distance from every set position in the pattern to the given move.

    Args:
        pattern: A pattern of pieces to measure against the move, of type fourbynine_pattern
        move: A move index to measure against the pattern, of type int (move.board_position).

    Returns:
        The average L1 distance between every piece in the given pattern and the given move, or 0 if no
        pieces are in the given pattern.
    """
    board_positions = pattern_string_to_board_positions(pattern.to_string())
    if not board_positions:
        return 0
    total_distance = 0
    move_coordinates = index_to_coordinate(move)
    for p in board_positions:
        total_distance += manhattan_distance(move_coordinates,
                                             index_to_coordinate(p))
    return float(total_distance) / len(board_positions)


def count_neighbors_of_move_in_pattern(pattern, move):
    """
    Calculates the number of orthogonal neighbors that the given move has in the given pattern.

    Args:
        pattern: A pattern of pieces to check against the move, of type fourbynine_pattern
        move: A move index to check against, of type int (move.board_position).

    Returns:
        The number of neighbors in pattern orthogonal to move.
    """
    board_positions = pattern_string_to_board_positions(pattern.to_string())
    if not board_positions:
        return 0
    num_neighbors = 0
    for p in board_positions:
        if manhattan_distance(index_to_coordinate(p), index_to_coordinate(move)) == 1:
            num_neighbors += 1
    return num_neighbors


def center_of_mass_of_pattern(pattern):
    """
    Calculates the center of mass of the pattern in row, col coordinates, where pieces
    are interpreted as having uniform density.

    Args:
        pattern: A pattern of pieces.

    Returns:
        The center of mass of the given pattern, i.e. the average of row and col across
        all pieces.
    """
    board_positions = pattern_string_to_board_positions(pattern.to_string())
    if not board_positions:
        return None
    return np.mean(np.array(list(map(index_to_coordinate, board_positions))), axis=0)


def check_board_for_new_patterns(board, move, pattern_checker):
    """
    Given a board and a move, checks to see if the given move produced any new near-win patterns along
    the horizontal, vertical, and diagonal axis that the move was played. This function uses a passed-in
    function, pattern_checker, to evaluate whether or not the given move produced a new pattern; if a new pattern
    is detected by pattern_checker, this function will immediately return True without evaluating any other patterns.
    This function will evaluate all possible 4-in-a-row positions containing the given move using the pattern_checker
    function.

    Args:
        board: The board the move is about to be played on.
        move: The move that is to be played.
        pattern_checker: A function that takes the _new_ board state (with the given move played), as well as a pattern
                         masking out a potential win. If the locations masked out by the pattern contain a pattern of
                         interest, pattern_checker should return True, at which point this function will also immediately
                         return True. If pattern_checker returns False, we continue to evaluate all possible win positions
                         containing the given move.

    Returns:
        True if pattern_checker detects an interesting pattern in any possible winning position containing the given move
        played against the given board, else False.
    """
    active_player = board.active_player()
    new_board = board + \
        fourbynine.fourbynine_move(int(move), 0.0, active_player)
    row, col = index_to_coordinate(move)

    def create_empty_pattern():
        return np.zeros(new_board.get_board_size()).reshape((new_board.get_board_height(), new_board.get_board_width()))

    # Generate all vertical 4-in-a-rows that contain the given move. There is only one possible solution.
    vertical = create_empty_pattern()
    vertical[:, col] = 1
    vertical_pattern = array_to_pattern(vertical)
    if pattern_checker(new_board, vertical_pattern):
        return True

    # Generate all horizontal 4-in-a-rows that contain the given move. There are up to 4 possible solutions.
    for i in range(-3, 1):
        horizontal = create_empty_pattern()
        starting_index = col + i
        if starting_index >= 0 and starting_index + 4 <= new_board.get_board_width():
            horizontal[row, starting_index:starting_index + 4] = 1
            horizontal_pattern = array_to_pattern(horizontal)
            if pattern_checker(new_board, horizontal_pattern):
                return True

    # Generate all diagonal 4-in-a-rows that contain the given move. There are up to 2 possible solutions.
    positive_diagonal_starting_index = col - row
    if positive_diagonal_starting_index >= 0 and positive_diagonal_starting_index <= new_board.get_board_width() - 4:
        positive_diagonal = create_empty_pattern()
        for i in range(0, new_board.get_board_height()):
            positive_diagonal[i, positive_diagonal_starting_index + i] = 1
        positive_diagonal_pattern = array_to_pattern(positive_diagonal)
        if pattern_checker(new_board, positive_diagonal_pattern):
            return True
    negative_diagonal_starting_index = col + row
    if negative_diagonal_starting_index >= 3 and negative_diagonal_starting_index < new_board.get_board_width():
        negative_diagonal = create_empty_pattern()
        for i in range(0, new_board.get_board_height()):
            negative_diagonal[i, negative_diagonal_starting_index - i] = 1
        negative_diagonal_pattern = array_to_pattern(negative_diagonal)
        if pattern_checker(new_board, negative_diagonal_pattern):
            return True
    return False


def plot_statistic(ax, moves, heuristic, statistic_function, y_axis_label, include_random=True):
    """
    Generates a plot on the given axis summarizing a particular board statistic. The x-axis is always
    move number, and the y-axis is determined by the passed in function, statistic_function.

    Args:
        ax: The axis to plot on.
        moves: A list of moves over which statistics will be calculated. This list can be generated by the output of
               parse_participant_file.
        heuristic: A heuristic to use as the baseline model for generating statistics. This heuristic will be sampled
                   for moves in all of the given positions in order to compare against the supplied move dataset.
        statistic_function: A function that takes in a board and a move index and returns the desired statistic to be summarized.
                            All output statistics should be floats, and are averaged across move numbers in order to produce the
                            final plots.
        y_axis_label: The name of this statistic, and the label given to the y-axis.
        include_random: If True, include a random move baseline to compare against.

    Returns:
        None, but adds a plot to the given axis with the output as a side effect.
    """
    board_size = fourbynine.fourbynine_board().get_board_size()

    # The index here is the number of pieces on the board.
    move_totals = np.zeros(board_size, dtype=int)
    model_statistics = np.zeros(board_size)
    player_statistics = np.zeros(board_size)
    if include_random:
        random_statistics = np.zeros(board_size)
    print("Calculating statistic: {}".format(y_axis_label))
    for move in tqdm(moves):
        def construct_move_histogram_from_position(board, heuristic, num_samples=200):
            position_counts = [0] * board.get_board_size()
            for i in range(num_samples):
                bfs = NInARowBestFirstSearch.create()
                best_move = heuristic.get_best_move(
                    board, board.active_player(), bfs)
                position_counts[best_move.board_position] += 1
            return position_counts

        histogram = construct_move_histogram_from_position(
            move.board, heuristic)
        best_move_position = np.argmax(histogram)

        board = move.board
        num_pieces = board.num_pieces()
        move_totals[num_pieces] += 1
        model_statistics[num_pieces] += statistic_function(
            board, best_move_position)
        player_statistics[num_pieces] += statistic_function(
            board, move.move.board_position)
        if include_random:
            random_move = heuristic.get_random_move(board)
            random_statistics[num_pieces] += statistic_function(
                board, random_move.board_position)

    def normalize(statistic):
        return np.nan_to_num(statistic / move_totals)

    # Plot
    ax.plot(np.arange(board_size), normalize(model_statistics),
            lw=2, marker='o', color='darkblue', label='Model')
    ax.plot(np.arange(board_size), normalize(player_statistics),
            lw=2, marker='o', color='darkorange', label='Data')
    if include_random:
        ax.plot(np.arange(board_size), normalize(random_statistics),
                lw=2, marker='o', color='darkgreen', label='Random')
    ax.set_xlabel('Number of pieces')
    ax.set_ylabel(y_axis_label)
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--participant_file",
        help="The file containing a list of positions to analyze.",
        required=True,
        type=str)
    parser.add_argument(
        "-p",
        "--params",
        help="The file containing a list of parameters for the model to compare to.",
        required=True,
        type=str)
    args = parser.parse_args()
    moves = parse_participant_file(args.participant_file)
    params = parse_bads_parameter_file_to_model_parameters(args.params)
    heuristic = fourbynine.fourbynine_heuristic.create(
        fourbynine.DoubleVector(params), True)

    fig, ax = plt.subplots(3, 3, figsize=(4, 4))

    def distance_from_center(_, move):
        return manhattan_distance(index_to_coordinate(move), np.array([1.5, 4]))

    plot_statistic(ax[0, 0], moves, heuristic,
                   distance_from_center, "Distance to\nboard center")

    def distance_from_own_pieces(board, move):
        return average_manhattan_distance_from_pattern(board.get_pieces(board.active_player()), move)

    plot_statistic(ax[0, 1], moves, heuristic,
                   distance_from_own_pieces, "Distance to\nown pieces")

    def distance_from_opponent_pieces(board, move):
        return average_manhattan_distance_from_pattern(board.get_pieces(fourbynine.get_other_player(board.active_player())), move)

    plot_statistic(ax[0, 2], moves, heuristic,
                   distance_from_opponent_pieces, "Distance to\nopponent's pieces")

    def distance_from_center_of_mass_of_own_pieces(board, move):
        return manhattan_distance(index_to_coordinate(move), center_of_mass_of_pattern(board.get_pieces(board.active_player())))

    plot_statistic(ax[1, 0], moves, heuristic,
                   distance_from_center_of_mass_of_own_pieces, "Distance to\nown center of mass")

    def distance_from_center_of_mass_of_opponent_pieces(board, move):
        return manhattan_distance(index_to_coordinate(move), center_of_mass_of_pattern(board.get_pieces(fourbynine.get_other_player(board.active_player()))))

    plot_statistic(ax[1, 1], moves, heuristic, distance_from_center_of_mass_of_opponent_pieces,
                   "Distance to\nopponent's center of mass")

    def number_of_own_orthogonal_neighbors(board, move):
        return count_neighbors_of_move_in_pattern(board.get_pieces(board.active_player()), move)

    plot_statistic(ax[1, 2], moves, heuristic,
                   number_of_own_orthogonal_neighbors, "Number of\nown neighbors")

    def number_of_opponent_orthogonal_neighbors(board, move):
        return count_neighbors_of_move_in_pattern(board.get_pieces(fourbynine.get_other_player(board.active_player())), move)

    plot_statistic(ax[2, 0], moves, heuristic,
                   number_of_own_orthogonal_neighbors, "Number of\nopponent's neighbors")

    def number_of_threats_made(board, move):
        def pattern_contains_threat(new_board, pattern):
            # Since this is the new board state after our move, the active player is actually no longer us - we need to check our own pieces, so we use get_other_player.
            return new_board.count_pieces(pattern, fourbynine.get_other_player(new_board.active_player())) == 3 and new_board.count_spaces(pattern) == 1

        if check_board_for_new_patterns(board, move, pattern_contains_threat):
            return 1.0
        return 0.0

    plot_statistic(ax[2, 1], moves, heuristic,
                   number_of_threats_made, "Number of\nthreats made")

    def number_of_threats_defended(board, move):
        def pattern_contains_defense(new_board, pattern):
            # This is the board state after our move, so the active player is the opponent and we are the other player.
            return new_board.count_pieces(pattern, new_board.active_player()) == 3 and new_board.count_pieces(pattern, fourbynine.get_other_player(new_board.active_player())) == 1

        if check_board_for_new_patterns(board, move, pattern_contains_defense):
            return 1.0
        return 0.0

    plot_statistic(ax[2, 2], moves, heuristic,
                   number_of_threats_defended, "Number of\nthreats defended")

    plt.show()


if __name__ == "__main__":
    main()
