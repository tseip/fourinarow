from fourbynine import *
from model_fit import bool_to_player, player_to_string, player_to_bool
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QRadioButton, QCheckBox, QGridLayout, QLineEdit, QLabel, QSplitter, QStyleFactory
from PyQt6.QtGui import QColor, QPalette
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


def pattern_string_to_board_positions(pattern_string):
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output


class HeuristicParameters(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setWindowTitle("Heuristic Parameter Editor")
        parameter_names = ["Stopping threshold", "Pruning threshold", "Gamma",
                           "Lapse rate", "Opponent scale", "Exploration constant", "Center weight"]
        for i in range(17):
            parameter_names.append("FP" + str(i) + " c_act")
        for i in range(17):
            parameter_names.append("FP" + str(i) + " c_pass")
        for i in range(17):
            parameter_names.append("FP" + str(i) + " delta")
        layout = QGridLayout()
        i = 0
        num_columns = 5
        for name in parameter_names:
            self._addParameter(layout, name, 0.0, i //
                               num_columns, i % num_columns)
            i += 1

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button, i // num_columns +
                         1, 0, 1, 2*num_columns)

        default_button = QPushButton("Reset to defaults")
        default_button.clicked.connect(self.reset_to_defaults)
        layout.addWidget(default_button, i // num_columns +
                         2, 0, 1, 2*num_columns)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.reset_to_defaults()

    def _addParameter(self, layout, parameter_name, parameter, i, j):
        label = QLabel(parameter_name)
        box = QLineEdit()
        box.setText(str(parameter))
        layout.addWidget(label, i, 2*j)
        layout.addWidget(box, i, 2*j+1)

    def get_params(self):
        params = []
        for box in self.findChildren(QLineEdit):
            params.append(float(box.displayText()))
        return params

    def save(self):
        self.parent.update_heuristic_parameters()
        self.hide()

    def reset_to_defaults(self):
        default_params = DoubleVector(DefaultFourByNineParameters)
        for box, param in zip(self.findChildren(QLineEdit), default_params):
            box.setText(str(param))

    def open_edit(self):
        self.show()


class BoardDisplay(QWidget):
    def __init__(self, feature_list, feature_list_toggle):
        super().__init__()
        self.board = fourbynine_board(
            fourbynine_pattern(0b0), fourbynine_pattern(0b0))
        self.fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.fig.mpl_connect('button_release_event', self.onclick)
        self.ax = self.fig.figure.add_subplot(111, aspect='equal')
        self.bfs = NInARowBestFirstSearch_create()
        self.iteration = 0
        self.heuristic_values = []
        self.heuristic_view = True
        self.parameter_editor = HeuristicParameters(self)
        self.feature_list = feature_list
        self.feature_list_toggle = feature_list_toggle

        self.hover = None

        splitter = QSplitter(Qt.Orientation.Vertical)

        board_widget = QWidget()
        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(self.fig)
        mpl_layout.addWidget(NavigationToolbar(self.fig, self))
        board_widget.setLayout(mpl_layout)

        middle_widget = QWidget()
        checkboxes = QHBoxLayout()
        self.ghost_check = QCheckBox("Show ghost")
        self.ghost_check.setChecked(True)

        play_move_button = QPushButton("Play move")
        play_move_button.clicked.connect(self.play_best_move)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)

        checkboxes.addWidget(self.ghost_check)
        checkboxes.addWidget(play_move_button)
        checkboxes.addWidget(reset_button)
        middle_widget.setLayout(checkboxes)

        bottom_widget = QWidget()
        seed_layout = QHBoxLayout()
        self.noise_check = QCheckBox("Enable noise")
        self.noise_check.setChecked(False)
        self.noise_check.stateChanged.connect(self.update_heuristic_parameters)
        seed_label = QLabel("Seed")
        self.seed_box = QLineEdit("0")
        seed_button = QPushButton("Save seed")
        seed_button.clicked.connect(self.update_heuristic_parameters)
        seed_layout.addWidget(self.noise_check)
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_box)
        seed_layout.addWidget(seed_button)
        bottom_widget.setLayout(seed_layout)

        button = QPushButton("Edit heuristic parameters")
        button.clicked.connect(self.parameter_editor.open_edit)

        splitter.addWidget(board_widget)
        splitter.addWidget(HeuristicViewToggleRadio(self))
        splitter.addWidget(middle_widget)
        splitter.addWidget(bottom_widget)
        splitter.addWidget(button)

        splitter.setHandleWidth(10)

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.update_heuristic_parameters()

    def onclick(self, event):
        if not event.xdata or not event.ydata:
            return
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if (event.button == 1):
            new_move = fourbynine_move(
                row, col, 0.0, self.board.active_player())
            self.play_move(new_move)
        elif (event.button == 3):
            old_move = fourbynine_move(
                row, col, 0.0, get_other_player(self.board.active_player()))
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
        self.ax.vlines(np.arange(-0.5, 9.5, 1), -0.5, 3.5)
        self.ax.hlines(np.arange(-0.5, 4.5, 1), -0.5, 8.5)

        def plot_circle(position, color, alpha=1.0):
            circ = patches.Circle((position % 9, position//9), 0.33,
                                  color=color, fill=True, alpha=alpha)
            circ = self.ax.add_patch(circ)
        pieces = set()
        for p in pattern_string_to_board_positions(self.board.get_pieces(Player_Player1).to_string()):
            plot_circle(p, "black")
            pieces.add(p)
        for p in pattern_string_to_board_positions(self.board.get_pieces(Player_Player2).to_string()):
            plot_circle(p, "white")
            pieces.add(p)

        if self.hover is not None and self.hover not in pieces and self.ghost_check.isChecked() and not self.board.game_has_ended():
            plot_circle(self.hover, player_to_string(
                self.board.active_player()), 0.5)

        if self.feature_list_toggle.isChecked():
            for p in pattern_string_to_board_positions(self.feature_list.pieces):
                plot_circle(p, "darkgreen", 0.8)
            for p in pattern_string_to_board_positions(self.feature_list.spaces):
                plot_circle(p, "darkred", 0.8)

        move_values = np.zeros(shape=[4, 9])
        if self.heuristic_view:
            for move in self.heuristic_values:
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
        self.ax.imshow(move_values, cmap=cm, vmin=-norm, vmax=norm,
                       interpolation='nearest', origin='upper')
        self.ax.axis('off')
        self.fig.figure.canvas.draw()

    def reset(self):
        self.board.reset()
        self.on_board_update()

    def play_move(self, new_move):
        if (self.board.contains_spaces(new_move.board_position) and not self.board.game_has_ended()):
            self.board += new_move
            self.on_board_update()

    def play_best_move(self):
        self.play_move(
            self.heuristic.get_best_known_move_from_search_tree(self.bfs))

    def on_board_update(self):
        self.heuristic_values = self.heuristic.get_moves(
            self.board, self.board.active_player())
        self.iteration = 0
        self.hover = None
        self.bfs.begin_search(
            self.heuristic, self.board.active_player(), self.board)
        self.feature_list.update(
            self.heuristic.get_feature_packs(), self.board)

    def dispatch(self):
        self.iteration += 1
        self.bfs.dispatch()
        if (self.bfs.root_valid()):
            self.candidate_moves = self.bfs.get_tree().get_children()

    def update_heuristic_parameters(self):
        self.heuristic = fourbynine_heuristic_create(
            DoubleVector(self.parameter_editor.get_params()))
        self.heuristic.seed_generator(int(self.seed_box.displayText()))
        self.heuristic.set_noise_enabled(self.noise_check.isChecked())
        self.on_board_update()


class MoveListItem(QListWidgetItem):
    def __init__(self, move):
        super().__init__(move.to_string())
        self.board_position = move.get_move().board_position


class MoveList(QListWidget):
    def __init__(self, board_view):
        super().__init__()

        self.setSortingEnabled(False)
        self.itemDoubleClicked.connect(self._onClick)
        self.currentItemChanged.connect(self._itemChanged)
        self.board_view = board_view

    def _onClick(self, item):
        new_move = fourbynine_move(
            item.board_position, 0.0, self.board_view.board.active_player())
        self.board_view.play_move(new_move)

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
        reverse = not player_to_bool(self.board_view.board.active_player())
        for move in sorted(moves, key=fourbynine_game_tree_node.get_value, reverse=reverse):
            self.addItem(MoveListItem(move))
        for i in range(self.count()):
            if (self.item(i).board_position == currentPosition):
                self.setCurrentRow(i)
                break


class FeatureListItem(QListWidgetItem):
    def __init__(self, label, feature):
        super().__init__(label)
        self.pieces = feature.pieces.to_string()
        self.spaces = feature.spaces.to_string()
        self.min_space_occupancy = feature.min_space_occupancy


class FeatureList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setSortingEnabled(False)
        self.currentItemChanged.connect(self._itemChanged)
        self.pieces = fourbynine_pattern().to_string()
        self.spaces = fourbynine_pattern().to_string()

    def _itemChanged(self, oldItem, newItem):
        if newItem is not None:
            self.pieces = newItem.pieces
            self.spaces = newItem.spaces

    def update(self, feature_packs, board):
        self.clear()
        for i, feature_pack in enumerate(feature_packs):
            label = "w_act: {}, w_pass: {}, delta: {}".format(
                feature_pack.weight_act, feature_pack.weight_pass, feature_pack.drop_rate)
            for j, feature in enumerate(feature_pack.features):
                if feature.is_active(board, board.active_player()):
                    self.addItem(FeatureListItem("fp {}, idx {}: ".format(
                        i, j) + label + " " + feature.to_string(), feature))


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
        splitter = QSplitter()

        feature_list = FeatureList()

        feature_list_toggle = QCheckBox("Show feature overlay")
        self.board = BoardDisplay(feature_list, feature_list_toggle)

        splitter.addWidget(self.board)

        self._timer = self.board.fig.new_timer(50)
        self._timer.add_callback(self._update_board)
        self._timer.start()

        self._move_timer = self.board.fig.new_timer(500)
        self._move_timer.add_callback(self._update_move_list)
        self._move_timer.start()

        move_list_layout = QVBoxLayout()
        move_list_layout.addWidget(QLabel("Search results"))
        self.move_list = MoveList(self.board)
        move_list_layout.addWidget(self.move_list)
        move_list_widget = QWidget()
        move_list_widget.setLayout(move_list_layout)
        splitter.addWidget(move_list_widget)

        feature_list_layout = QVBoxLayout()
        feature_list_layout.addWidget(QLabel("Active features"))
        feature_list_layout.addWidget(feature_list)
        feature_list_layout.addWidget(feature_list_toggle)
        feature_list_widget = QWidget()
        feature_list_widget.setLayout(feature_list_layout)
        splitter.addWidget(feature_list_widget)

        self.setCentralWidget(splitter)

        self.setMinimumSize(QSize(400, 300))

    def _update_board(self):
        self.board.dispatch()
        self.board.show()

    def _update_move_list(self):
        self.move_list._update(self.board.candidate_moves)


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
