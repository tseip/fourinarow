from fourbynine import *
from model_fit import bool_to_player, player_to_string
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np

def pattern_to_cells(pattern):
    pattern_string = pattern.to_string()
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[i] == "1":
            output.append(i)
    return output

class BoardDisplay:
    def __init__(self, board, heuristic):
        self.board = board
        self.heuristic = heuristic
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('button_release_event', self.onclick)
        self.ax = self.fig.add_subplot(111,aspect='equal')

    def onclick(self, event):
        if not event.xdata or not event.ydata:
            return
        col, row = 9 - int(event.xdata + 0.5) - 1, 4 - int(event.ydata + 0.5) - 1
        if (event.button == 1):
            new_move = fourbynine_move(row, col, 0.0, self.board.active_player())
            if (self.board.contains_spaces(new_move.board_position)):
                self.board += new_move
        elif (event.button == 3):
            old_move = fourbynine_move(row, col, 0.0, get_other_player(self.board.active_player()))
            if (self.board.contains_move(old_move)):
                self.board -= old_move
        self.show()
        
    def show(self):
        """
        Code to display a board as a matplotlib figure
        """

        cm = colors.LinearSegmentedColormap.from_list('red_green_map', [colors.to_rgb('tab:red'), 
                                                                    colors.to_rgb('lightgray'), colors.to_rgb('tab:green')], N=100)
        cm.set_over(color='k')
        cm.set_under(color='w')

        self.ax.clear()
        self.ax.vlines(np.arange(-0.5,9.5,1),-0.5,3.5)
        self.ax.hlines(np.arange(-0.5,4.5,1),-0.5,8.5)
        for p in pattern_to_cells(self.board.get_pieces(Player_Player1)):
            circ = patches.Circle((p%9,p//9),0.33,color="black",fill=True)
            circ = self.ax.add_patch(circ)
        for p in pattern_to_cells(self.board.get_pieces(Player_Player2)):
            circ = patches.Circle((p%9,p//9),0.33,color="white",fill=True)
            circ = self.ax.add_patch(circ)

        move_values = np.zeros(shape=[4,9])
        if (self.heuristic):
            for move in self.heuristic.get_moves(self.board, self.board.active_player()):
                col = 9 - (move.board_position % 9) - 1
                row = 4 - (move.board_position // 9) - 1
                move_values[row][col] = move.val
        norm = np.max(np.abs(move_values[np.isfinite(move_values)])) + 0.001
        move_values[np.isposinf(move_values)] = norm + 1
        move_values[np.isneginf(move_values)] = -norm - 1
        plt.imshow(move_values, cmap=cm, vmin=-norm, vmax=norm, interpolation='nearest', origin='upper')
        self.ax.axis('off')
        plt.show()
    

def main():
    position = fourbynine_board(
        fourbynine_pattern(0b1), fourbynine_pattern(0b0))
    heuristic = getDefaultFourByNineHeuristic()
    heuristic.seed_generator(random.randint(0, 2**64))
    #heuristic.set_noise_enabled(False)
    bd = BoardDisplay(position, heuristic)
    bd.show()

if __name__ == "__main__":
    main()
