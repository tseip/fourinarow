from fourbynine import *
from parsers import parse_participant_file, parse_bads_parameter_file_to_model_parameters
import argparse
import random
from tqdm import tqdm


def search_from_position(position, heuristic, noise_enabled=True):
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch.create()
    bfs.search(heuristic, position.active_player(), position)
    return bfs.get_tree()


def sample_planning_depth(heuristic, positions, num_samples, disable_tqdm=True):
    total_depth = 0
    for position in tqdm(positions, disable=disable_tqdm):
        for i in range(num_samples):
            total_depth += search_from_position(position,
                                                heuristic).get_depth_of_pv()
    return float(total_depth) / (len(positions) * num_samples)


def sample_average_branching_factor(heuristic, positions, num_samples, disable_tqdm=True):
    total_branching_factor = 0.0
    for position in tqdm(positions, disable=disable_tqdm):
        for i in range(num_samples):
            total_branching_factor += search_from_position(
                position, heuristic).get_average_branching_factor()
    return float(total_branching_factor) / (len(positions) * num_samples)


def calculate_tree_statistics_from_file(path, heuristic, num_samples=10):
    moves = parse_participant_file(path)
    positions = [move.board for move in moves]
    return sample_planning_depth(heuristic, positions, num_samples, False), sample_average_branching_factor(heuristic, positions, num_samples, False)


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
        help="The file containing the parameters for the model.",
        type=str)
    args = parser.parse_args()
    if args.params:
        params = parse_bads_parameter_file_to_model_parameters(args.params)
        heuristic = fourbynine_heuristic.create(DoubleVector(params), True)
    else:
        heuristic = fourbynine_heuristic.create()
    print("Average planning depth: {}, Average branching factor: {}".format(
        *calculate_tree_statistics_from_file(args.participant_file, heuristic)))


if __name__ == "__main__":
    main()
