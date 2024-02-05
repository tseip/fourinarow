from fourbynine import *
from calculate_tree_statistics import search_from_position, sample_planning_depth, sample_average_branching_factor
from feature_utilities import *
import random
import time


class ModifiedBestFirstSearch(NInARowBestFirstSearch):
    def __init__(self, heuristic, board):
        super().__init__(heuristic, board)

    def select_next_node(self):
        return super().select_next_node()

    def stopping_conditions(self, heuristic, board):
        return super().stopping_conditions(heuristic, board)

    def on_node_expansion(self, expanded_node, heuristic, board):
        return super().on_node_expansion(expanded_node, heuristic, board)


def move_to_csv_string(board, move, time, group_id, participant):
    def board_to_base10_str(board):
        return str(int(board.to_string(), 2))
    return "\t".join([board_to_base10_str(board.get_pieces(bool_to_player(False))), board_to_base10_str(board.get_pieces(bool_to_player(True))), player_to_string(move.player), str(2**move.board_position), str(time), str(group_id), str(participant)]) + "\n"


def create_feature(white_pieces, black_pieces, min_occupancy):
    return fourbynine_heuristic_feature(fourbynine_pattern(white_pieces), fourbynine_pattern(black_pieces), min_occupancy)


def create_custom_heuristic():
    # Creating a heuristic with a single feature group
    heuristic_params = DoubleVector([7.0, 5.0, 0.01, 0.01, 1.0, 0.0, 1.0])
    heuristic = fourbynine_heuristic.create(heuristic_params, False)
    heuristic.add_feature_group(0.8, 0.8, 0.2)
    heuristic.add_feature(0, create_feature(0x3, 0xc, 2))
    heuristic.add_feature(0, create_feature(0x600, 0x1800, 2))
    return heuristic


def modify_default_heuristic():
    # Taking the default heuristic and adding a new group of features as well
    # as modifying an old feature and an old feature group.
    heuristic = fourbynine_heuristic.create()
    feature_group_weights = heuristic.get_feature_group_weights()
    features = heuristic.get_features_with_metadata()

    # Replace a feature and remove a feature
    features[0].feature = create_feature(0x1, 0x2, 2)

    # Move this feature to group 1
    features[0].group_index = 1

    # Change the weighting of all features in group 0.
    feature_group_weights[0].weight_act = 0.75

    # Erase the second feature.
    features.erase(features.begin() + 1)

    # Disable the second feature group entirely
    feature_group_weights[1].weight_act = 0.0
    feature_group_weights[1].weight_pass = 0.0
    feature_group_weights[1].drop_rate = 1.0

    # Add a new feature group
    heuristic.add_feature_group(0.1, 0.2, 0.3)
    heuristic.add_feature(17, create_feature(0x3, 0xc, 2))
    return heuristic


def create_triangle_heuristic():
    # Taking the default heuristic and adding a new group of features.
    heuristic = fourbynine_heuristic.create()

    # Add a new feature group. These weights are arbitrary.
    heuristic.add_feature_group(0.8, 0.2, 0.2)

    # Construct the features we want.
    triangle_features = []
    triangle_features.append(create_feature(
        0b100000000110000000000, 0b1100000000000000001001000000101, 3))
    triangle_features.append(create_feature(
        0b010000001100000000000, 0b1000001100000000011000001000, 3))
    triangle_features.append(create_feature(
        0b1010000000100000000000, 0b100000000100000001010000001010, 3))

    for triangle_feature in triangle_features:
        for feature in generate_feature_transformations(triangle_feature, True, True):
            heuristic.add_feature(17, feature)
    return heuristic


def evaluate_best_move_from_position(position, noise_enabled=True):
    # Get the default heuristic and disable noise so we can
    # see what the heuristic actually encodes
    heuristic = fourbynine_heuristic.create()
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch(heuristic, position)
    bfs.complete_search()
    best_move = heuristic.get_best_move(bfs.get_tree())
    return best_move


def play_game_to_completion(heuristic):
    # Play an entire game with noise enabled.
    moves = []
    heuristic.seed_generator(random.randint(0, 2**64))
    current_player = False
    current_position = fourbynine_board()
    while not current_position.game_has_ended():
        start = time.time()
        bfs = NInARowBestFirstSearch(
            heuristic, current_position)
        bfs.complete_search()
        best_move = heuristic.get_best_move(bfs.get_tree())
        end = time.time()
        moves.append(move_to_csv_string(current_position,
                     best_move, end - start, 1, "DefaultHeuristic"))
        current_position = current_position + best_move
        current_player = not current_player
    return moves, current_position


def main():
    create_custom_heuristic()
    modify_default_heuristic()

    heuristic = fourbynine_heuristic.create()
    position = fourbynine_board(
        fourbynine_pattern(0x0), fourbynine_pattern(0x0))
    best_starting_move = evaluate_best_move_from_position(position, False)
    print("Best starting move:")
    print((position + best_starting_move).to_string())
    root = search_from_position(position, heuristic, False)
    print("Average planning depth:")
    print(sample_planning_depth(heuristic, [position], 100))
    print("Average branching factor:")
    print(sample_average_branching_factor(heuristic, [position], 100))
    unpacked_evaluations = map(lambda x: (
        x.get_move().board_position, x.get_value()), root.get_children())
    print("BFS Search heuristic evaluations for all possible starting moves:")
    print(sorted(list(unpacked_evaluations), key=lambda x: x[1], reverse=True))
    output_csv = []
    idealized_game, final_position = play_game_to_completion(heuristic)
    print(final_position.to_string())
    for line in idealized_game:
        print(line)
    # We could output our game to a CSV file like so:
    # with open("example_output.csv", 'w') as f:
    #     f.writelines(idealized_game)

    triangle_position = fourbynine_board(fourbynine_pattern(
        0b000000000000010000000011000000000000), fourbynine_pattern(0b111000000000000000000000000000000000))
    # Counting the number of active features of a given position
    triangle_heuristic = create_triangle_heuristic()
    print("Active feature counts per pack for black: {}".format(
        count_features(triangle_heuristic, triangle_position, Player_Player1)))
    print("Active feature counts per pack for white: {}".format(
        count_features(triangle_heuristic, triangle_position, Player_Player2)))
    search = ModifiedBestFirstSearch(
        heuristic, position)


if __name__ == "__main__":
    main()
