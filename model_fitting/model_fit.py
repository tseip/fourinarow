from collections import defaultdict
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import random
import fourbynine
import copy
import time
from queue import Empty, Full, PriorityQueue
from multiprocessing import Process, Pool, Value, Lock, Queue
from multiprocessing.managers import SyncManager
from pybads import BADS
from pathlib import Path
from functools import total_ordering

expt_factor = 1.0
cutoff = 3.5

x0 = np.array([2.0, 0.02, 0.2, 0.05, 1.2, 0.8,
              1, 0.4, 3.5, 5], dtype=np.float64)
ub = np.array([10.0, 1, 1, 1, 4, 10, 10, 10, 10, 10], dtype=np.float64)
lb = np.array([0.1, 0.001, 0, 0, 0.25, -10, -
              10, -10, -10, -10], dtype=np.float64)
pub = np.array([9.99, 0.99, 0.5, 0.5, 2, 5, 5, 5, 5, 5], dtype=np.float64)
plb = np.array([1, 0.1, 0.001, 0.05, 0.5, -5, -
               5, -5, -5, -5], dtype=np.float64)
c = 50


def bool_to_player(player):
    return fourbynine.Player_Player1 if not player else fourbynine.Player_Player2


def player_to_bool(player):
    return player == fourbynine.Player_Player2


def player_to_string(player):
    return "White" if player_to_bool(player) else "Black"


def board_position_to_tile_number(x):
    return ((1 + (x ^ (x-1))) >> 1).bit_length() - 1


@total_ordering
class CSVMove:
    def __init__(
            self,
            black_pieces,
            white_pieces,
            player,
            move,
            time,
            group,
            participant):
        self.black_pieces = int(black_pieces)
        self.white_pieces = int(white_pieces)
        self.player = bool(player)
        self.move = int(move)
        self.time = float(time)
        self.group = int(group)
        self.participant = str(participant)

    def __hash__(self):
        return hash(
            (self.black_pieces,
             self.white_pieces,
             self.player,
             self.move,
             self.time,
             self.participant))

    def __repr__(self):
        return "\t".join([str(self.black_pieces), str(self.white_pieces), player_to_string(self.player), str(self.move), str(self.time), str(self.group), self.participant])

    def __eq__(self, other):
        return self.black_pieces == other.black_pieces and self.white_pieces == other.white_pieces and self.player == other.player and self.move == other.move and self.time == other.time and self.group == other.group and self.participant == other.participant

    def __lt__(self, other):
        return (self.black_pieces, self.white_pieces, self.player, self.move) < (other.black_pieces, other.white_pieces, other.player, other.move)


class SuccessFrequencyTracker:
    def __init__(
            self):
        self.attempt_count = 1
        self.success_count = 0
        self.required_success_count = 1
        self.L = 0.0

    def __repr__(self):
        return " ".join([str(self.attempt_count), str(self.success_count), str(self.required_success_count)])

    def is_done(self):
        return self.success_count == self.required_success_count

    def report_success(self, success):
        if success:
            self.success_count += 1
            if not self.is_done():
                self.attempt_count = 1
        else:
            self.L += expt_factor / \
                (self.required_success_count * self.attempt_count)
            self.attempt_count += 1


def parse_participant_lines(lines):
    moves = []
    for line in lines:
        # Try splitting by comma
        parameters = line.rstrip().split(',')
        if (len(parameters) == 1):
            parameters = line.rstrip().split()
        if (len(parameters) == 6):
            moves.append(CSVMove(
                parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], 1, parameters[5]))
        elif (len(parameters) == 7):
            moves.append(CSVMove(parameters[0], parameters[1], True if parameters[2].lower(
            ) == "white" else False, parameters[3], parameters[4], parameters[5], parameters[6]))
        else:
            raise Exception(
                "Given input has incorrect number of parameters (expected 6 or 7): " + line)
    return moves


def generate_splits(moves, split_count):
    indices = list(range(len(moves)))
    if (split_count != 1):
        random.shuffle(indices)
    count = 0
    groups = []
    for i in range(split_count):
        groups.append([])
    for i in range(len(moves)):
        groups[i % split_count].append(moves[indices[i]])
        groups[i % split_count][-1].group = (i % split_count) + 1
    return groups


def estimate_log_lik_ibs(
        parameters,
        input_queue,
        output_queue):
    heuristic = fourbynine.getDefaultFourByNineHeuristic(
        fourbynine.DoubleVector(parameters))
    heuristic.seed_generator(random.randint(0, 2**64))
    bfs = fourbynine.NInARowBestFirstSearch.create()
    while True:
        move = input_queue.get()
        b = fourbynine.fourbynine_pattern(move.black_pieces)
        w = fourbynine.fourbynine_pattern(move.white_pieces)
        board = fourbynine.fourbynine_board(b, w)
        player = bool_to_player(move.player)
        best_move = heuristic.get_best_move(board, player, bfs)
        success = 2**best_move.board_position == move.move
        output_queue.put((success, move))


def compute_loglik(move_tasks, params):
    move_tasks = copy.deepcopy(move_tasks)
    moves_to_process = len(move_tasks)
    N = moves_to_process
    Lexpt = N * expt_factor
    num_workers = 20
    submission_queue = Queue(num_workers)
    results_queue = Queue(num_workers)
    to_submit_queue = PriorityQueue()
    pool = Pool(num_workers, estimate_log_lik_ibs,
                (params, submission_queue, results_queue,))
    for move in move_tasks:
        to_submit_queue.put((0, move))
    while moves_to_process and Lexpt <= cutoff * N:
        # Feed the furnace
        while not submission_queue.full():
            try:
                submission_count, move = to_submit_queue.get_nowait()
            except Empty:
                break
            try:
                submission_queue.put_nowait(move)
                submission_count += 1
            except Full:
                break
            finally:
                if not move_tasks[move].is_done():
                    to_submit_queue.put((submission_count, move))
        # Process results
        while not results_queue.empty():
            try:
                success, move = results_queue.get_nowait()
                if not move_tasks[move].is_done():
                    if (success):
                        Lexpt -= expt_factor / \
                            move_tasks[move].required_success_count
                    else:
                        Lexpt += expt_factor / \
                            (move_tasks[move].required_success_count *
                             move_tasks[move].attempt_count)
                    move_tasks[move].report_success(success)
                    if move_tasks[move].is_done():
                        moves_to_process -= 1
            except Empty:
                break
    pool.terminate()
    pool.join()

    L_values = {}
    for move in move_tasks:
        L_values[move] = move_tasks[move].L
    return L_values


def generate_attempt_counts(L_values, c):
    x = np.linspace(1e-6, 1 - 1e-6, int(1e6))
    dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
    p = np.exp(-L_values)
    interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
    interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
    times = (c * interp1(p)) / np.mean(interp2(p))
    return np.vectorize(lambda x: max(x, 1))(np.round(times))


def parse_parameters(params):
    if (len(params) != 10):
        raise Exception("Parameter file must contain 10 parameters!")
    out = [10000, params[0], params[1], params[3], 1, 1, params[5]]
    out.extend([x for x in params[6:]] * 4)
    out.append(0)
    out.extend([x * params[4] for x in params[6:]] * 4)
    out.append(0)
    out.extend([params[2]] * 17)
    return out


def fit_model(moves, verbose=False):
    move_tasks = {}
    for move in moves:
        move_tasks[move] = SuccessFrequencyTracker()
    if verbose:
        print("Beginning model fit pre-processing: log-likelihood estimation")
    l_values = []
    for i in range(10):
        if verbose:
            print("Theta:", x0)
        l_values.append(compute_loglik(move_tasks, parse_parameters(x0)))
    average_l_values = []
    for move in moves:
        average = 0.0
        for l_value in l_values:
            average += l_value[move]
        average /= len(l_values)
        average_l_values.append(average)
    counts = generate_attempt_counts(np.array(average_l_values), c)
    for i in range(len(counts)):
        move_tasks[moves[i]].required_success_count = int(counts[i])

    def opt_fun(x):
        if verbose:
            print("Theta:", x)
        return sum(list(compute_loglik(move_tasks, parse_parameters(x)).values()))
    badsopts = {}
    badsopts['uncertainty_handling'] = True
    badsopts['noise_final_samples'] = 0
    badsopts['max_fun_evals'] = 2000
    bads = BADS(opt_fun, x0, lb, ub, plb, pub, options=badsopts)
    out_params = bads.optimize()['x']
    l_values = []
    for i in range(10):
        l_values.append(opt_fun(out_params))
    return out_params, l_values


def cross_validate(groups, i):
    test = groups[i]
    train = []
    if len(groups) == 1:
        train.extend(groups[0])
    else:
        for j in range(len(groups)):
            if i != j:
                train.extend(groups[j])
    params, loglik_train = fit_model(train)
    test_tasks = {}
    for move in test:
        test_tasks[move] = SuccessFrequencyTracker()
    loglik_test = list(compute_loglik(
        test_tasks, parse_parameters(params)).values())
    return params, loglik_train, loglik_test


def main():
    random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--participant_file",
        help="The file containing participant data to be split, i.e. a list of board states, moves, and associated timing. Optionally, a number of splits may be provided if cross-validation is desired.",
        nargs='+',
        metavar=(
            'input_file',
            'split_count'))
    parser.add_argument(
        "-i",
        "--input_dir",
        help="The directory containing the pre-split groups to parse and cross-validate, along with the expected number of splits to be parsed. These splits should be named [1-n].csv",
        nargs=2,
        metavar=(
            'input_dir',
            'split_count'))
    parser.add_argument(
        "output_dir",
        help="The directory to output results to.",
        type=str,
        default="./")
    args = parser.parse_args()
    if args.participant_file and args.input_dir:
        raise Exception("Can't specify both -f and -i!")

    groups = []
    if args.participant_file:
        num_splits = 1
        if (len(args.participant_file) == 2):
            num_splits = int(args.participant_file[1])
        if (len(args.participant_file) > 2):
            raise Exception("-f only takes up to 2 arguments!")
        with open(args.participant_file[0], 'r') as lines:
            moves = parse_participant_lines(lines)
        groups = generate_splits(moves, num_splits)
    elif args.input_dir:
        input_path = Path(args.input_dir[0])
        num_splits = int(args.input_dir[1])
        input_files = []
        for i in range(num_splits):
            input_files.append(input_path / (str(i + 1) + ".csv"))
        for input_path in input_files:
            with input_path.open('r') as lines:
                groups.append(parse_participant_lines(lines))
    else:
        raise Exception("Either -f or -i must be specified!")
    output_path = Path(args.output_dir)
    if not output_path.is_dir():
        output_path.mkdir()
    for i in range(len(groups)):
        with (output_path / (str(i + 1) + ".csv")).open('w') as f:
            for move in groups[i]:
                f.write(str(move) + "\n")

    for i in range(len(groups)):
        params, loglik_train, loglik_test = cross_validate(groups, i)
        with (output_path / ("params" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in params))
        with (output_path / ("lltrain" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in loglik_train))
        with (output_path / ("lltest" + str(i + 1) + ".csv")).open('w') as f:
            f.write(' '.join(str(x) for x in loglik_test) + '\n')


if __name__ == "__main__":
    main()
