from fourbynine import *
from model_fit import bool_to_player
import random


def create_feature(white_pieces, black_pieces, min_occupancy):
    return fourbynine_heuristic_feature(fourbynine_pattern(white_pieces), fourbynine_pattern(black_pieces), min_occupancy)


def create_custom_heuristic():
    # Creating a heuristic with a single feature pack
    heuristic_params = DoubleVector([7.0, 5.0, 0.01, 0.01, 1.0, 0.0, 1.0])
    heuristic = fourbynine_heuristic(heuristic_params)
    heuristic.add_feature_pack(0.8, 0.8, 0.2)
    heuristic.add_feature(0, create_feature(0x3, 0xc, 2))
    heuristic.add_feature(0, create_feature(0x600, 0x1800, 2))
    return heuristic


def modify_default_heuristic():
    # Taking the default heuristic and adding a new pack of features as well
    # as modifying an old feature and an old feature pack.
    heuristic = getDefaultFourByNineHeuristic()
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


def evaluate_best_move_from_position(position):
    # Get the default heuristic and disable noise so we can
    # see what the heuristic actually encodes
    heuristic = getDefaultFourByNineHeuristic()
    heuristic.set_noise_enabled(False)
    best_move = heuristic.get_best_move_bfs(position, Player_Player1)
    return best_move


def play_game_to_completion():
    # Play an entire game with noise enabled.
    heuristic = getDefaultFourByNineHeuristic()
    heuristic.seed_generator(random.randint(0, 2**64))
    current_player = False
    current_position = fourbynine_board()
    while not current_position.game_has_ended():
        current_position = current_position + \
            heuristic.get_best_move_bfs(
                current_position, bool_to_player(current_player))
        current_player = not current_player
    return current_position


def main():
    create_custom_heuristic()
    modify_default_heuristic()

    position = fourbynine_board(
        fourbynine_pattern(0x0), fourbynine_pattern(0x0))
    best_starting_move = evaluate_best_move_from_position(position)
    print((position + best_starting_move).to_string())
    idealized_game = play_game_to_completion()
    print(idealized_game.to_string())


if __name__ == "__main__":
    main()
