from fourbynine import *
from model_fit import bool_to_player, player_to_string
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QRadioButton, QCheckBox
from PyQt6.QtGui import QColor, QPalette
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


def pattern_to_cells(pattern):
    pattern_string = pattern.to_string()
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output

class BoardDisplay:
    def __init__(self, heuristic):
        self.board = fourbynine_board(fourbynine_pattern(0b0), fourbynine_pattern(0b0))
        self.heuristic = heuristic
        self.fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.fig.mpl_connect('button_release_event', self.onclick)
        self.ax = self.fig.figure.add_subplot(111, aspect='equal')
        self.bfs = NInARowBestFirstSearch_create()
        self.iteration = 0
        self.on_board_update()
        self.hover = None
        self.heuristic_view = True
        
    def onclick(self, event):
        if not event.xdata or not event.ydata:
            return
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if (event.button == 1):
            new_move = fourbynine_move(row, col, 0.0, self.board.active_player())
            if (self.board.contains_spaces(new_move.board_position) and not self.board.game_has_ended()):
                self.board += new_move
                self.on_board_update()
        elif (event.button == 3):
            old_move = fourbynine_move(row, col, 0.0, get_other_player(self.board.active_player()))
            if (self.board.contains_move(old_move)):
                self.board -= old_move
                self.on_board_update()
        
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

        if self.hover is not None:
            color = player_to_string(self.board.active_player())
            circ = patches.Circle((self.hover%9,self.hover//9),0.33,color=color,fill=True, alpha=0.5)
            circ = self.ax.add_patch(circ)
            
        move_values = np.zeros(shape=[4,9])
        if self.heuristic_view:
            for move in self.heuristic.get_moves(self.board, self.board.active_player()):
                col = (move.board_position % 9)
                row = (move.board_position // 9)
                move_values[row][col] = move.val
        else:
            for move in self.candidate_moves:
                col = (move.get_move().board_position % 9)
                row = (move.get_move().board_position // 9)
                move_values[row][col] = move.get_value()
        norm = np.max(np.abs(move_values[np.isfinite(move_values)])) + 0.001
        move_values[np.isposinf(move_values)] = norm + 1
        move_values[np.isneginf(move_values)] = -norm - 1
        self.ax.imshow(move_values, cmap=cm, vmin=-norm, vmax=norm, interpolation='nearest', origin='upper')
        self.ax.axis('off')
        self.fig.figure.canvas.draw()

    def on_board_update(self):
        self.iteration = 0
        self.bfs.begin_search(self.heuristic, self.board.active_player(), self.board)

    def dispatch(self):
        self.iteration += 1
        self.bfs.dispatch()
        if (self.bfs.root_valid()):
            self.candidate_moves = self.bfs.get_tree().get_children()

class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)
        self.setMinimumSize(100,100)

class MoveListItem(QListWidgetItem):
    def __init__(self, move):
        super().__init__(move.to_string())
        self.board_position = move.get_move().board_position
        
class MoveList(QListWidget):
    def __init__(self, board_view):
        super().__init__()
        self.setSortingEnabled(False)
        self.setWindowTitle("example")
        self.itemDoubleClicked.connect(self._onClick)
        self.currentItemChanged.connect(self._itemChanged)
        self.board_view = board_view
    
    def _onClick(self, item):
        print("wtf")

    def _itemChanged(self, oldItem, newItem):
        if newItem is not None:
            self.board_view.hover = newItem.board_position

    def _update(self, moves):
        currentItem = self.currentItem()
        if currentItem:
            currentPosition = currentItem.board_position
        else:
            currentPosition = 0
        self.clear()
        for move in sorted(moves, key=fourbynine_game_tree_node.get_value, reverse=True):
            self.addItem(MoveListItem(move))
        for i in range(self.count()):
            if (self.item(i).board_position == currentPosition):
                self.setCurrentRow(i)
                break

class HeuristicViewToggleRadio(QWidget):
    def __init__(self, board_view):
      super().__init__()
      
      layout = QHBoxLayout()
      self.b1 = QRadioButton("1-ply Heuristic")
      self.b1.heuristic = True
      self.b1.setChecked(True)
      self.b1.toggled.connect(self._on_toggle)
      layout.addWidget(self.b1)
		
      self.b2 = QRadioButton("Search Heuristic")
      self.b2.toggled.connect(self._on_toggle)
      self.b2.heuristic = False

      layout.addWidget(self.b2)
      self.setLayout(layout)
      self.board_view = board_view

    def _on_toggle(self):
        radioButton = self.sender()
        if (radioButton.isChecked()):
            self.board_view.heuristic_view = radioButton.heuristic
            
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("4IAR Game Explorer")
        layout = QHBoxLayout()

        mpl_layout = QVBoxLayout()
        heuristic = getDefaultFourByNineHeuristic()
        heuristic.seed_generator(random.randint(0, 2**64))
        heuristic.set_noise_enabled(False)

        self.board = BoardDisplay(heuristic)
        mpl_layout.addWidget(self.board.fig)
        mpl_layout.addWidget(NavigationToolbar(self.board.fig, self))
        mpl_layout.addWidget(HeuristicViewToggleRadio(self.board))
        mpl_layout.addWidget(QCheckBox("Show ghost"))
        
        layout.addLayout(mpl_layout)
        
        self._timer = self.board.fig.new_timer(50)
        self._timer.add_callback(self._update_board)
        self._timer.start()

        self._move_timer = self.board.fig.new_timer(500)
        self._move_timer.add_callback(self._update_move_list)
        self._move_timer.start()

        self.move_list = MoveList(self.board)
        layout.addWidget(self.move_list)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setMinimumSize(QSize(400, 300))

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def _update_board(self):
        self.board.dispatch()
        #self.move_list._update(self.board.candidate_moves)
        self.board.show()

    def _update_move_list(self):
        self.move_list._update(self.board.candidate_moves)
        
def main():
    heuristic = getDefaultFourByNineHeuristic()
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(False)
    app = QApplication([])
    window = MainWindow()
    window.show()

    app.exec()
    #bd = BoardDisplay(position, heuristic)
    #bd.show()

if __name__ == "__main__":
    main()
