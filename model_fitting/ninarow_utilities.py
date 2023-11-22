import numpy as np
import argparse
import random
from pathlib import Path
from fourbynine import NInARowBestFirstSearch


def search_from_position(position, heuristic, noise_enabled=True, seed=None):
    if seed:
        heuristic.seed_generator(seed)
    else:
        heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch(heuristic, position.active_player(), position)
    bfs.complete_search()
    return bfs.get_tree()


def bads_parameters_to_model_parameters(params):
    if (len(params) != 10):
        raise Exception(
            "Parameter file must contain 10 parameters: {}".format(params))
    params = list(map(float, params))
    out = [10000.0, params[0], params[1], params[3], 1, 1, params[5]]
    out.extend([x for x in params[6:]] * 4)
    out.append(0)
    out.extend([x * params[4] for x in params[6:]] * 4)
    out.append(0)
    out.extend([params[2]] * 17)
    return out


def get_heuristic_quality(params):
    """
    Given model parameters (of specifically length 58), evaluate the correlation of the parameters with a pre-derived set
    of optimal parameters.

    Args:
        params: The parameters to evaluate

    Returns:
        A number in the range [-1, 1] representing the correlation of the given parameters with the optimal parameters.
    """
    feature_counts = np.loadtxt(
        Path('heuristic_quality_inputs/optimal_feature_vals.txt'))[:, -35:]

    optimal_move_values = np.loadtxt(
        Path('heuristic_quality_inputs/opt_hvh.txt'))[:, -36:]

    # columns are player id, color, cross-validation group, number of pieces, chosen move, and response time in ms
    move_stats_hvh = np.loadtxt(
        Path('heuristic_quality_inputs/move_stats_hvh.txt'), dtype=int)
    num_pieces_hvh = move_stats_hvh[:, 3]

    mask = ~np.isnan(optimal_move_values)
    optimal_move_values[mask] = np.vectorize(
        lambda x: -1 if x < -5000 else (1 if x > 5000 else 0))(optimal_move_values[mask])

    player_color = move_stats_hvh[:, 1]
    optimal_board_values = np.full_like(
        player_color, fill_value=np.nan, dtype=float)
    optimal_board_values[player_color == 0] = np.nanmax(
        optimal_move_values[player_color == 0, :], axis=1)
    optimal_board_values[player_color == 1] = - \
        np.nanmin(optimal_move_values[player_color == 1, :], axis=1)

    params = np.array(params)
    f3inarow = (params[9]+params[28])/2
    heuristic_values = np.tanh(0.4*np.sum((-2*player_color+1)[:, None]*feature_counts
                                          * params[None, 6:41]/f3inarow, axis=1))
    return np.corrcoef(heuristic_values, optimal_board_values)[0, 1]


def main():
    parser = argparse.ArgumentParser(
        description="If passed a parameters file, parses the parameters and evaluates them against optimal parameters. Returns correlation with optimal parameters.")
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="The file containing the parameters for the model.",
        type=str)
    args = parser.parse_args()
    from parsers import parse_bads_parameter_file_to_model_parameters
    params = parse_bads_parameter_file_to_model_parameters(args.params)
    print(get_heuristic_quality(params))


if __name__ == "__main__":
    main()
