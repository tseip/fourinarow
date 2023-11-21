from fourbynine import *
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np


def pattern_string_to_board_positions(pattern_string):
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output


class BoardRenderer():
    def __init__(self, figure, ax):
        self.fig = figure
        self.ax = ax

        self.clear()

    def add_piece(self, position, color, alpha=1.0):
        if position is not None:
            piece = patches.Circle((position % fourbynine_board().get_board_width(), position//fourbynine_board().get_board_width()), 0.33,
                                   color=color, fill=True, alpha=alpha)
            self.ax.add_patch(piece)
            self.pieces.add(position)
            self.draw()

    def add_ghost(self, position, color):
        if position not in self.pieces:
            self.add_piece(position, color, 0.5)

    def set_board(self, board):
        self.clear()
        self.board = board

        self.ax.vlines(np.arange(-0.5, self.board.get_board_width() +
                       0.5, 1), -0.5, self.board.get_board_height() - 0.5)
        self.ax.hlines(np.arange(-0.5, self.board.get_board_height() +
                       0.5, 1), -0.5, self.board.get_board_width() - 0.5)
        for p in pattern_string_to_board_positions(self.board.get_pieces(Player_Player1).to_string()):
            self.add_piece(p, "black")
        for p in pattern_string_to_board_positions(self.board.get_pieces(Player_Player2).to_string()):
            self.add_piece(p, "white")
        self.set_position_values([])
        self.draw()

    def set_position_values(self, position_values):
        move_values = np.zeros(shape=[4, 9])
        for position in position_values:
            move_values[position[0]][position[1]] = position_values[position]
        norm = np.max(np.abs(move_values[np.isfinite(move_values)])) + 0.001
        move_values[np.isposinf(move_values)] = norm + 1
        move_values[np.isneginf(move_values)] = -norm - 1

        cm = colors.LinearSegmentedColormap.from_list('red_green_map', [colors.to_rgb('tab:red'),
                                                                        colors.to_rgb('lightgray'), colors.to_rgb('tab:green')], N=100)
        cm.set_over(color='k')
        cm.set_under(color='w')
        self.ax.imshow(move_values, cmap=cm, vmin=-norm, vmax=norm,
                       interpolation='nearest', origin='upper')
        self.draw()

    def clear(self):
        self.ax.clear()
        self.ax.axis('off')
        self.board = None
        self.pieces = set()
        self.draw()

    def draw(self):
        self.fig.figure.canvas.draw()


def main():
    fig, ax = plt.subplots()
    renderer = BoardRenderer(fig, ax)
    renderer.set_board(fourbynine_board(
        fourbynine_pattern(1), fourbynine_pattern(0b10)))
    renderer.draw()
    plt.show()


if __name__ == "__main__":
    main()
