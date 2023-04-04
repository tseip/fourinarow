from collections import defaultdict
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import random
import fourbynine
from multiprocessing import Process, Pool, Value, JoinableQueue, Lock
from pybads import BADS

expt_factor = 1.0
cutoff = 3.5

x0 = np.array([   2, 0.02,  0.2,0.05, 1.2,0.8,  1,0.4,3.5,  5])
ub = np.array([  10,    1,    1,   1,   4, 10, 10, 10, 10, 10])
lb = np.array([   0,    0,    0,0.00,0.25,-10,-10,-10,-10,-10])
pub = np.array([ 9.99, 0.99,  0.5, 0.5,   2,  5,  5,  5,  5,  5])
plb = np.array([0.1, 0.001,0.001,0.05, 0.5, -5, -5, -5, -5, -5])
c = 50

class MoveEvaluationTask:
    def __init__(self, black_pieces, white_pieces, player, move, time, participant):
        self.black_pieces = black_pieces
        self.white_pieces = white_pieces
        self.player = player
        self.move = move
        self.time = time
        self.participant = participant
        self.attempt_count = 1
        self.success_count = 0
        self.required_success_count = 1
        self.L = 0.0

    def __repr__(self):
        return '{' + ' '.join(map(str, [self.black_pieces, self.white_pieces, self.player, self.move, self.time, self.participant])) + '}'

    def __eq__(self, other):
        return self.black_pieces == other.black_pieces and self.white_pieces == other.white_pieces and self.player == other.player and self.move == other.move and self.time == other.time and self.participant == other.participant
    
    def __hash__(self):
        return hash((self.black_pieces, self.white_pieces, self.player, self.move, self.time, self.participant))

    def is_done(self):
        return self.success_count == self.required_success_count
    
    def report_success(self, success):
        if success:
            self.success_count += 1
            if not self.is_done():
                self.attempt_count = 1
        else:
            self.L += expt_factor / (self.required_success_count * self.attempt_count)
            self.attempt_count += 1

def bool_to_player(player):
    return fourbynine.Player_Player1 if not player else fourbynine.Player_Player2
    
def parse_participant_lines(lines):
    moves = []
    for line in lines:
        parameters = line.rstrip().split(',')
        if (len(parameters) != 6):
            raise Exception("Given input has incorrect number of parameters (expected 6): " + line)
        moves.append(MoveEvaluationTask(int(parameters[0]), int(parameters[1]), bool(parameters[2]), int(parameters[3]), float(parameters[4]), parameters[5]))
    return moves

def estimate_log_lik_ibs(participant_data, parameters, Lexpt, to_process, cutoff, output):
    heuristic = fourbynine.fourbynine_heuristic(fourbynine.FourByNineFeatures)
    seed = 2**64
    heuristic.seed_generator(random.randint(0, seed))
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
                    Lexpt.value += expt_factor / (move.required_success_count * move.attempt_count)
            move.report_success(success)
            if not move.is_done():
                participant_data.put(move)
            else:
                with to_process.get_lock():
                    to_process.value -= 1
                output.put(move)
            participant_data.task_done()
        except:
            pass

def calc_L_values(moves, params):
    q = JoinableQueue()
    for move in moves:
        q.put(move)

    N = len(moves)
    output = JoinableQueue(N)
    Lexpt = Value('d', N * expt_factor, lock=True)
    to_process = Value('i', N, lock=True)
    num_workers = 16
    pool = Pool(num_workers, estimate_log_lik_ibs, (q, params, Lexpt, to_process, cutoff * N, output,))
    pool.close()
    pool.join()
    q.close()
    q.join_thread()

    final_L_values = {}
    while not output.empty():
        move = output.get()
        final_L_values[move] = move.L
    output.close()
    output.join_thread()    
    output_L_values = []
    for move in moves:
        output_L_values.append(final_L_values[move])
    return output_L_values

def generate_attempt_counts(L_values, c):
    x = np.linspace(1e-6, 1-1e-6, int(1e6))
    dilog = np.pi**2/6.0 + np.cumsum(np.log(x)/(1-x))/len(x)
    p = np.exp(-L_values)
    interp1 = CubicSpline(x, np.sqrt(x*dilog), extrapolate=True)
    interp2 = CubicSpline(x, np.sqrt(dilog/x), extrapolate=True)
    times = (c * interp1(p)) / np.mean(interp2(p))
    return np.vectorize(lambda x: max(x, 1))(np.int32(times))
        
def parse_parameters(params):
    if (len(params) != 10):
        raise Exception("Parameter file must contain 10 parameters!")
    out = [10000, params[0], params[1], params[3], 1, 1, params[5]]
    out.extend(params[6:]*4)
    out.append(0)
    out.extend([x * params[4] for x in params[6:]]*4)
    out.append(0)
    out.extend([params[2]]*17)
    return out
        
def main():
    random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("participant_file", help="The file containing participant data, i.e. a list of board states, moves, and associated timing.", type=str)
    args = parser.parse_args()
    with open(args.participant_file, 'r') as lines:
        moves = parse_participant_lines(lines)
    l_values = []
    for i in range(10):
        l_values.append(np.array(calc_L_values(moves, parse_parameters(x0))))
    average_l_values = np.mean(l_values, axis=0)
    counts = generate_attempt_counts(average_l_values, c)
    for i in range(len(counts)):
        moves[i].required_success_count = counts[i]
    opt_fun = lambda x: sum(calc_L_values(moves, parse_parameters(x)))
    badsopts = {}
    badsopts['uncertainty_handling'] = True
    badsopts['noise_final_samples'] = 0
    badsopts['max_fun_evals'] = 2000
    print(x0)
    print(lb)
    print(ub)
    print(plb)
    print(pub)
    bads = BADS(opt_fun, x0, lb, ub, plb, pub, options=badsopts)
    optimized_params = bads.optimize()
    print(optimized_params)
    
if __name__ == "__main__":
    main()
