from collections import defaultdict
from functools import total_ordering
from UltraDict import UltraDict
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import random
import fourbynine
import copy
import time
from multiprocessing import Process, Pool, Value, Lock
from pybads import BADS
from pathlib import Path
from tqdm import tqdm
from parsers import *


class SuccessFrequencyTracker:
    def __init__(
            self, expt_factor):
        self.attempt_count = 1
        self.success_count = 0
        self.required_success_count = 1
        self.L = 0.0
        self.expt_factor = expt_factor

    def __repr__(self):
        return " ".join([str(self.attempt_count), str(self.success_count), str(self.required_success_count)])

    def is_done(self):
        return self.success_count == self.required_success_count

    def report_success(self, success):
        if success:
            self.success_count += 1
            if not self.is_done():
                self.attempt_count = 1
            return -(self.expt_factor / self.required_success_count)
        else:
            delta = self.expt_factor / \
                (self.required_success_count * self.attempt_count)
            self.L += delta
            self.attempt_count += 1
            return delta


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


class DefaultModel:
    def __init__(self):
        self.expt_factor = 1.0
        self.cutoff = 3.5

        self.x0 = np.array([2.0, 0.02, 0.2, 0.05, 1.2, 0.8,
                            1, 0.4, 3.5, 5], dtype=np.float64)
        self.ub = np.array(
            [10.0, 1, 1, 1, 4, 10, 10, 10, 10, 10], dtype=np.float64)
        self.lb = np.array([0.1, 0.001, 0, 0, 0.25, -10, -
                            10, -10, -10, -10], dtype=np.float64)
        self.pub = np.array([9.99, 0.99, 0.5, 0.5, 2, 5,
                            5, 5, 5, 5], dtype=np.float64)
        self.plb = np.array([1, 0.1, 0.001, 0.001, 0.5, -5, -
                             5, -5, -5, -5], dtype=np.float64)
        self.c = 50

    def create_heuristic(self, params):
        return fourbynine.fourbynine_heuristic.create(fourbynine.DoubleVector(bads_parameters_to_model_parameters(params)), True)

    def create_search(self, params, heuristic, board):
        return fourbynine.NInARowBestFirstSearch(heuristic, board)

    def estimate_initial_l_value_guess(self, fitter, moves):
        return fitter.estimate_l_values(moves, self.x0, 10)


class ModelFitter:
    def __init__(self, model, args):
        self.model = model
        if args.random_sample:
            self.random_sample = True
            self.sample_count = args.random_sample[0]
            if self.sample_count <= 1:
                raise Exception("Sample count must be greater than one!")
        else:
            self.random_sample = False
            self.sample_count = 0
        self.verbose = args.verbose

    def estimate_log_lik_ibs(
            self,
            params,
            cutoff,
            move_tasks):
        heuristic = self.model.create_heuristic(params)
        heuristic.seed_generator(random.randint(0, 2**64))
        while Lexpt.value <= cutoff:
            unfinished_items = list(
                filter(lambda x: not x[1].is_done(), move_tasks.items()))
            if not unfinished_items:
                break
            move, task = copy.deepcopy(random.choice(unfinished_items))
            local_Lexpt_delta = 0
            while not move_tasks[move].is_done():
                search = self.model.create_search(
                    params, heuristic, move.board)
                search.complete_search()
                best_move = heuristic.get_best_move(search.get_tree())
                success = best_move.board_position == move.move.board_position
                local_Lexpt_delta += task.report_success(success)
                if (success):
                    with move_tasks.lock:
                        if task.success_count == move_tasks[move].success_count + 1:
                            move_tasks[move] = task
                            Lexpt.value += local_Lexpt_delta
                    break
                else:
                    # We may need to exit early. This implicitly signals all of the other processes as well.
                    if Lexpt.value + local_Lexpt_delta > cutoff:
                        with move_tasks.lock:
                            Lexpt.value += local_Lexpt_delta
                        break

    def compute_loglik(self, move_tasks, params):
        global pool, Lexpt
        move_tasks = copy.deepcopy(move_tasks)
        N = len(move_tasks)
        num_workers = 16

        cutoff = N * self.model.cutoff
        Lexpt.value = N * self.model.expt_factor
        shared_tasks = UltraDict(move_tasks)
        results = [pool.apply_async(
            self.estimate_log_lik_ibs, (params, cutoff, shared_tasks,)) for i in range(num_workers)]
        [result.wait() for result in results]

        L_values = {}
        for move in shared_tasks:
            L_values[move] = shared_tasks[move].L
        return L_values

    def generate_attempt_counts(self, L_values, c):
        x = np.linspace(1e-6, 1 - 1e-6, int(1e6))
        dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
        p = np.exp(-L_values)
        interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
        interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
        times = (c * interp1(p)) / np.mean(interp2(p))
        return np.vectorize(lambda x: max(x, 1))(np.round(times))

    def estimate_l_values(self, moves, params, sample_count):
        move_tasks = {}
        for move in moves:
            move_tasks[move] = SuccessFrequencyTracker(self.model.expt_factor)
        l_values = []
        for i in tqdm(range(sample_count)):
            l_values.append(self.compute_loglik(
                move_tasks, params))
        average_l_values = []
        for move in tqdm(moves):
            average = 0.0
            for l_value in l_values:
                average += l_value[move]
            average /= len(l_values)
            average_l_values.append(average)
        return average_l_values

    def fit_model(self, moves):
        print("Beginning model fit pre-processing: log-likelihood estimation")
        average_l_values = self.model.estimate_initial_l_value_guess(
            self, moves)
        counts = self.generate_attempt_counts(
            np.array(average_l_values), self.model.c)
        move_tasks = {}
        for move in moves:
            move_tasks[move] = SuccessFrequencyTracker(self.model.expt_factor)
        for i in range(len(counts)):
            move_tasks[moves[i]].required_success_count = int(counts[i])

        if self.random_sample:
            clamped_sample_count = min(self.sample_count, len(move_tasks))
            if clamped_sample_count != self.sample_count:
                print("Requested sample count ({}) is larger than the dataset size ({})! Clamping sample count and using entire set...".format(
                    self.sample_count, len(move_tasks)))

        def opt_fun(x):
            if self.verbose:
                print("Probing function at theta = {}".format(x))
                print("Current iteration: {}".format(
                    opt_fun.current_iteration_count))
                opt_fun.current_iteration_count += 1
            if self.random_sample:
                subsampled_keys = random.sample(
                    sorted(move_tasks), clamped_sample_count)
                subsampled_tasks = {k: move_tasks[k] for k in subsampled_keys}
            else:
                subsampled_tasks = move_tasks
            return sum(list(self.compute_loglik(subsampled_tasks, x).values()))

        opt_fun.current_iteration_count = 0
        badsopts = {}
        badsopts['uncertainty_handling'] = True
        badsopts['noise_final_samples'] = 0
        badsopts['max_fun_evals'] = 2000
        bads = BADS(opt_fun, self.model.x0, self.model.lb, self.model.ub,
                    self.model.plb, self.model.pub, options=badsopts)
        out_params = bads.optimize()['x']
        print("Final estimated params: {}".format(out_params))
        print("Beginning model fit post-processing: log-likelihood estimation")
        final_l_values = self.estimate_l_values(moves, out_params, 10)
        return out_params, final_l_values

    def cross_validate(self, groups, i):
        print("Cross validating split {} against the other {} splits".format(
            i + 1, len(groups) - 1))
        test = groups[i]
        train = []
        if len(groups) == 1:
            train.extend(groups[0])
        else:
            for j in range(len(groups)):
                if i != j:
                    train.extend(groups[j])
        params, loglik_train = self.fit_model(train)
        test_tasks = {}
        for move in test:
            test_tasks[move] = SuccessFrequencyTracker(self.model.expt_factor)
        loglik_test = list(self.compute_loglik(test_tasks, params).values())
        return params, loglik_train, loglik_test


def main():
    random.seed()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""Example usages:
Ingest a file named input.csv and output to a folder named output/: model_fit.py -f input.csv -o output/
Ingest a file, generate 5 splits, and cross validate: model_fit.py -f input.csv 5 -o output/
Generate 5 splits from a file and terminate: model_fit.py -f input.csv 5 -s -o output/
Read in splits from the above command and cross validate: model_fit.py -i output/ 5 -o output/
Read in splits from the above command, and only process/cross validate a single split (split 2, in this case) against the rest: model_fit.py -i output/ 5 -o output/ -c 2""")
    parser.add_argument(
        "-f",
        "--participant_file",
        help="The file containing participant data to be split, i.e. a list of board states, moves, and associated timing. Optionally, a number of splits may be provided if cross-validation is desired.",
        type=str,
        nargs='+',
        metavar=(
            'input_file',
            'split_count'))
    parser.add_argument(
        "-i",
        "--input_dir",
        help="The directory containing the pre-split groups to parse and cross-validate, along with the expected number of splits to be parsed. These splits should be named [1-n].csv",
        type=str,
        nargs=2,
        metavar=(
            'input_dir',
            'split_count'))
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The directory to output results to.",
        type=str,
        default="./",
        metavar=('output_dir'))
    parser.add_argument(
        "-s",
        "--splits-only",
        help="If specified, terminate after generating splits.",
        action='store_true')
    parser.add_argument(
        "-v",
        "--verbose",
        help="If specified, print extra debugging info.",
        action='store_true')
    parser.add_argument(
        "-c",
        "--cluster-mode",
        nargs=1,
        type=int,
        help="If specified, only process a single split, specified by the number passed as an argument to this flag. The split is expected to be named [arg].csv. This split will then be cross-validated against the other splits in the folder specified by the -i flag. Cannot be used with the -f flag; pre-split a -f argument with -s if desired.",
        metavar=('local_split'))
    parser.add_argument(
        "-r",
        "--random-sample",
        nargs=1,
        type=int,
        help="If specified, instead of testing each position on a BADS function evaluation, instead randomly sample up to N positions without replacement.",
        metavar=('sample_count'))
    parser.add_argument(
        "-t",
        "--threads",
        nargs=1,
        type=int,
        default=16,
        help="The number of threads to use when fitting.")
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
        if (args.cluster_mode):
            raise Exception("-c cannot be used with -f!")
        moves = parse_participant_file(args.participant_file[0])
        groups = generate_splits(moves, num_splits)
    elif args.input_dir:
        input_path = Path(args.input_dir[0])
        num_splits = int(args.input_dir[1])
        input_files = []
        for i in range(num_splits):
            input_files.append(input_path / (str(i + 1) + ".csv"))
        for input_path in input_files:
            print("Ingesting split {}".format(input_path))
            moves = parse_participant_file(input_path)
            groups.append(moves)
    else:
        raise Exception("Either -f or -i must be specified!")

    output_path = Path(args.output_dir)
    if not output_path.is_dir():
        output_path.mkdir()

    # Only output splits if we generated new ones to output.
    if args.participant_file:
        for i in range(len(groups)):
            new_split_path = output_path / (str(i + 1) + ".csv")
            print("Writing split {}".format(new_split_path))
            with (new_split_path).open('w') as f:
                for move in groups[i]:
                    f.write(str(move) + "\n")

    if args.splits_only:
        exit()

    global pool, Lexpt
    Lexpt = Value('d', 0)

    def thread_initializer(shared_val):
        global Lexpt
        Lexpt = shared_val
    pool = Pool(args.threads, initializer=thread_initializer,
                initargs=(Lexpt,))
    model_fitter = ModelFitter(DefaultModel(), args)
    start, end = 0, len(groups)
    if (args.cluster_mode):
        start = args.cluster_mode[0] - 1
        end = start + 1
    for i in range(start, end):
        params, loglik_train, loglik_test = model_fitter.cross_validate(
            groups, i)
        with (output_path / ("params" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in params))
        with (output_path / ("lltrain" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in loglik_train))
        with (output_path / ("lltest" + str(i + 1) + ".csv")).open('w') as f:
            f.write(' '.join(str(x) for x in loglik_test) + '\n')


if __name__ == "__main__":
    main()
