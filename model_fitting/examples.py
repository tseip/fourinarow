from fourbynine import *
from calculate_tree_statistics import search_from_position, sample_planning_depth, sample_average_branching_factor
from model_fit import bool_to_player, player_to_string
import random
import time


def move_to_csv_string(board, move, time, group_id, participant):
    def board_to_base10_str(board):
        return str(int(board.to_string(), 2))
    return "\t".join([board_to_base10_str(board.get_pieces(bool_to_player(False))), board_to_base10_str(board.get_pieces(bool_to_player(True))), player_to_string(move.player), str(2**move.board_position), str(time), str(group_id), str(participant)]) + "\n"


def create_feature(white_pieces, black_pieces, min_occupancy):
    return fourbynine_heuristic_feature(fourbynine_pattern(white_pieces), fourbynine_pattern(black_pieces), min_occupancy)


def create_custom_heuristic():
    # Creating a heuristic with a single feature pack
    heuristic_params = DoubleVector([7.0, 5.0, 0.01, 0.01, 1.0, 0.0, 1.0])
    heuristic = fourbynine_heuristic.create(heuristic_params, False)
    heuristic.add_feature_pack(0.8, 0.8, 0.2)
    heuristic.add_feature(0, create_feature(0x3, 0xc, 2))
    heuristic.add_feature(0, create_feature(0x600, 0x1800, 2))
    return heuristic


def modify_default_heuristic():
    # Taking the default heuristic and adding a new pack of features as well
    # as modifying an old feature and an old feature pack.
    heuristic = fourbynine_heuristic.create()
    feature_packs = heuristic.get_feature_packs()

    # Replace a feature in the first pack, and remove a feature
    feature_packs[0].features[0] = create_feature(0x1, 0x2, 2)
    feature_packs[0].weight_act = 0.75
    feature_packs[0].features.erase(feature_packs[0].features.begin() + 1)

    # Disable the second feature pack entirely
    feature_packs[1].weight_act = 0.0
    feature_packs[1].weight_pass = 0.0
    feature_packs[1].drop_rate = 1.0

    # Add a new feature pack
    heuristic.add_feature_pack(0.1, 0.2, 0.3)
    heuristic.add_feature(17, create_feature(0x3, 0xc, 2))
    return heuristic


def evaluate_best_move_from_position(position, noise_enabled=True):
    # Get the default heuristic and disable noise so we can
    # see what the heuristic actually encodes
    heuristic = fourbynine_heuristic.create()
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch.create()
    best_move = heuristic.get_best_move(
        position, position.active_player(), bfs)
    return best_move


def play_game_to_completion(heuristic):
    # Play an entire game with noise enabled.
    moves = []
    heuristic.seed_generator(random.randint(0, 2**64))
    current_player = False
    current_position = fourbynine_board()
    while not current_position.game_has_ended():
        start = time.time()
        bfs = NInARowBestFirstSearch.create()
        best_move = heuristic.get_best_move(
            current_position, bool_to_player(current_player), bfs)
        end = time.time()
        moves.append(move_to_csv_string(current_position,
                     best_move, end - start, 1, "DefaultHeuristic"))
        current_position = current_position + best_move
        current_player = not current_player
    return moves, current_position


def count_features(heuristic, position, player):
    feature_counts_per_pack = []
    for feature_pack in heuristic.get_feature_packs():
        feature_count = 0
        for feature in feature_pack.features:
            if feature.is_active(position, player) and feature.contained(position, player):
                feature_count += 1
        feature_counts_per_pack.append(feature_count)
    return feature_counts_per_pack


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

    # Counting the number of active features of a given position
    print("Active feature counts per pack: {}".format(
        count_features(heuristic, position, Player_Player1)))


if __name__ == "__main__":
    main()
