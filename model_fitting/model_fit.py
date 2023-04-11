from collections import defaultdict
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import random
import fourbynine
import copy
import queue
from multiprocessing import Process, Pool, Value, JoinableQueue, Lock
from pybads import BADS
from pathlib import Path

expt_factor = 1.0
cutoff = 3.5

x0 = np.array([2, 0.02, 0.2, 0.05, 1.2, 0.8, 1, 0.4, 3.5, 5])
ub = np.array([10, 1, 1, 1, 4, 10, 10, 10, 10, 10])
lb = np.array([0, 0, 0, 0.00, 0.25, -10, -10, -10, -10, -10])
pub = np.array([9.99, 0.99, 0.5, 0.5, 2, 5, 5, 5, 5, 5])
plb = np.array([0.1, 0.001, 0.001, 0.05, 0.5, -5, -5, -5, -5, -5])
c = 50


class MoveEvaluationTask:
    def __init__(
            self,
            black_pieces,
            white_pieces,
            player,
            move,
            time,
            group,
            participant):
        self.black_pieces = black_pieces
        self.white_pieces = white_pieces
        self.player = player
        self.move = move
        self.time = time
        self.group = group
        self.participant = participant
        self.attempt_count = 1
        self.success_count = 0
        self.required_success_count = 1
        self.L = 0.0

    def __repr__(self):
        player_string = "White" if self.player else "Black"
        return ' '.join(map(str,
                            [self.black_pieces,
                             self.white_pieces,
                             player_string,
                             self.move,
                             self.time,
                             self.group,
                             self.participant]))

    def __eq__(self, other):
        return self.black_pieces == other.black_pieces and self.white_pieces == other.white_pieces and self.player == other.player and self.move == other.move and self.time == other.time and self.participant == other.participant

    def __hash__(self):
        return hash(
            (self.black_pieces,
             self.white_pieces,
             self.player,
             self.move,
             self.time,
             self.participant))

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


def bool_to_player(player):
    return fourbynine.Player_Player1 if not player else fourbynine.Player_Player2


def parse_participant_lines(lines):
    moves = []
    for line in lines:
        # Try splitting by comma
        parameters = line.rstrip().split(',')
        if (len(parameters) == 1):
            parameters = line.rstrip().split()
        if (len(parameters) == 6):
            moves.append(MoveEvaluationTask(int(parameters[0]), int(parameters[1]), bool(
                parameters[2]), int(parameters[3]), float(parameters[4]), 1, parameters[5]))
        elif (len(parameters) == 7):
            moves.append(
                MoveEvaluationTask(
                    int(
                        parameters[0]), int(
                        parameters[1]), True if parameters[2].lower() == "white" else False, int(
                        parameters[3]), float(
                        parameters[4]), int(
                            parameters[5]), parameters[6]))
        else:
            raise Exception(
                "Given input has incorrect number of parameters (expected 6 or 7): " + line)
    return moves


def generate_splits(moves, split_count):
    indices = list(range(len(moves)))
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
        participant_data,
        parameters,
        Lexpt,
        to_process,
        cutoff,
        output):
    heuristic = fourbynine.getDefaultFourByNineHeuristic(
        fourbynine.DoubleVector(parameters))
    heuristic.seed_generator(random.randint(0, 2**64))
    while True:
        with Lexpt.get_lock():
            if Lexpt.value > cutoff:
                break
        with to_process.get_lock():
            if to_process.value == 0:
                break
        try:
            move = participant_data.get(False)
            b = fourbynine.fourbynine_pattern(move.black_pieces)
            w = fourbynine.fourbynine_pattern(move.white_pieces)
            board = fourbynine.fourbynine_board(b, w)
            player = bool_to_player(move.player)
            best_move = heuristic.get_best_move_bfs(board, player)
            success = 2**best_move.board_position == move.move
            with Lexpt.get_lock():
                if (success):
                    Lexpt.value -= expt_factor / move.required_success_count
                else:
                    Lexpt.value += expt_factor / \
                        (move.required_success_count * move.attempt_count)
            move.report_success(success)
            if not move.is_done():
                participant_data.put(move)
            else:
                with to_process.get_lock():
                    to_process.value -= 1
                output.put(move)
            participant_data.task_done()
        except queue.Empty:
            pass


def compute_loglik(moves, params):
    q = JoinableQueue()
    for move in moves:
        q.put(move)

    N = len(moves)
    output = JoinableQueue(N)
    Lexpt = Value('d', N * expt_factor, lock=True)
    to_process = Value('i', N, lock=True)
    num_workers = 16
    pool = Pool(num_workers, estimate_log_lik_ibs,
                (q, params, Lexpt, to_process, cutoff * N, output,))
    pool.close()
    pool.join()

    final_L_values = {}
    while not q.empty():
        move = q.get()
        final_L_values[move] = move.L
    q.close()
    q.join_thread()

    while not output.empty():
        move = output.get()
        final_L_values[move] = move.L
    output.close()
    output.join_thread()

    sorted_L_values = []
    for move in moves:
        sorted_L_values.append(final_L_values[move])
    return sorted_L_values


def generate_attempt_counts(L_values, c):
    x = np.linspace(1e-6, 1 - 1e-6, int(1e6))
    dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
    p = np.exp(-L_values)
    interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
    interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
    times = (c * interp1(p)) / np.mean(interp2(p))
    return np.vectorize(lambda x: max(x, 1))(np.int32(times))


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


def fit_model(data):
    print("Beginning model fit pre-processing: log-likelihood estimation")
    moves = copy.deepcopy(data)
    l_values = []
    for i in range(10):
        l_values.append(np.array(compute_loglik(moves, parse_parameters(x0))))
    average_l_values = np.mean(l_values, axis=0)
    counts = generate_attempt_counts(average_l_values, c)
    for i in range(len(counts)):
        moves[i].required_success_count = counts[i]

    def opt_fun(x): return sum(compute_loglik(moves, parse_parameters(x)))
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
        train = groups[0]
    else:
        for j in range(len(groups)):
            if i != j:
                train.extend(groups[j])
    params, loglik_train = fit_model(train)
    loglik_test = compute_loglik(test, parse_parameters(params))
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
