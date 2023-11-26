from fourbynine import *
from parsers import parse_participant_file, parse_bads_parameter_file_to_model_parameters
from ninarow_plotting import BoardRenderer, SearchRenderer
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QRadioButton, QCheckBox, QGridLayout, QLineEdit, QLabel, QSplitter, QStyleFactory, QComboBox, QFileDialog
from PyQt6.QtGui import QColor, QPalette
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import math


def pattern_string_to_board_positions(pattern_string):
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output


class TreeView(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.fig = FigureCanvas(Figure(figsize=(10, 10)))
        self.ax = self.fig.figure.add_subplot(111, aspect='auto')
        self.renderer = SearchRenderer(self.ax)
        self.fig.mpl_connect('button_press_event', self.renderer.onclick)

        def on_click_callback(boards):
            self.parent.move_history.splice_boards(boards)
        self.renderer.register_onclick_callback(on_click_callback)
        self.setWindowTitle("Tree View")

        tree_view_settings_widget = QWidget()
        tree_view_settings_layout = QHBoxLayout()
        tree_view_settings_layout.addWidget(QLabel("Max branching factor:"))
        self.max_branching_factor = QLineEdit(
            str(self.renderer.max_branching_factor))
        tree_view_settings_layout.addWidget(self.max_branching_factor)
        tree_view_settings_layout.addWidget(QLabel("Max depth:"))
        self.max_depth = QLineEdit(str(self.renderer.max_depth))
        tree_view_settings_layout.addWidget(self.max_depth)
        tree_view_settings_layout.addWidget(QLabel("Board size:"))
        self.board_size = QLineEdit(str(self.renderer.board_size))
        tree_view_settings_layout.addWidget(self.board_size)
        update_button = QPushButton("Update")
        update_button.clicked.connect(self.update)
        tree_view_settings_layout.addWidget(update_button)

        tree_view_settings_widget.setLayout(tree_view_settings_layout)

        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(self.fig)
        mpl_layout.addWidget(NavigationToolbar(self.fig, self))
        mpl_layout.addWidget(tree_view_settings_widget)
        widget = QWidget()
        widget.setLayout(mpl_layout)
        self.setCentralWidget(widget)

    def update(self):
        self.renderer.max_branching_factor = int(
            self.max_branching_factor.displayText())
        self.renderer.max_depth = int(self.max_depth.displayText())
        self.renderer.board_size = float(self.board_size.displayText())
        self.renderer.set_root(self.parent.search.get_tree())
        self.fig.figure.canvas.draw()

    def open_view(self):
        self.update()
        self.show()


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

        load_button = QPushButton("Load from file")
        load_button.clicked.connect(self.load_from_file)
        layout.addWidget(load_button, i // num_columns +
                         3, 0, 1, 2*num_columns)

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

    def set_params(self, params):
        for box, param in zip(self.findChildren(QLineEdit), params):
            box.setText(str(param))

    def save(self):
        self.parent.update_heuristic_parameters()
        self.hide()

    def reset_to_defaults(self):
        default_params = DoubleVector(DefaultFourByNineParameters)
        self.set_params(default_params)

    def open_edit(self):
        self.show()

    def load_from_file(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if len(filenames) > 1:
                raise Exception(
                    "Can only load parameters from one file at a time!")
                return
            if len(filenames) == 1:
                model_params = parse_bads_parameter_file_to_model_parameters(
                    filenames[0])
                self.set_params(model_params)


class BoardDisplay(QWidget):
    def __init__(self, feature_list, feature_list_toggle):
        super().__init__()
        self.board = fourbynine_board(
            fourbynine_pattern(0b0), fourbynine_pattern(0b0))
        self.heuristic = fourbynine_heuristic.create()
        self.search = self.create_search(
            self.heuristic, self.board.active_player(), self.board)
        self.fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.fig.mpl_connect('button_release_event', self.onclick)
        self.ax = self.fig.figure.add_subplot(111, aspect='equal')
        self.board_renderer = BoardRenderer(self.ax)
        self.heuristic_values = []
        self.heuristic_view = True
        self.tree_view = TreeView(self)
        self.parameter_editor = HeuristicParameters(self)
        self.feature_list = feature_list
        self.feature_list_toggle = feature_list_toggle
        self.move_history = MoveHistory(self)

        self.hover = None
        self.player_ghost = None

        splitter = QSplitter(Qt.Orientation.Vertical)

        board_widget = QWidget()
        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(self.fig)
        mpl_layout.addWidget(NavigationToolbar(self.fig, self))
        board_widget.setLayout(mpl_layout)

        middle_widget = QWidget()
        checkboxes = QHBoxLayout()
        self.ghost_check = QCheckBox("Show ghosts")
        self.ghost_check.setChecked(True)

        play_move_button = QPushButton("Play move")
        play_move_button.clicked.connect(self.play_best_move)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)

        checkboxes.addWidget(self.ghost_check)
        checkboxes.addWidget(play_move_button)
        checkboxes.addWidget(reset_button)
        middle_widget.setLayout(checkboxes)

        noise_widget = QWidget()
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
        noise_widget.setLayout(seed_layout)

        params_button = QPushButton("Edit heuristic parameters")
        params_button.clicked.connect(self.parameter_editor.open_edit)

        tree_button = QPushButton("Open tree view")
        tree_button.clicked.connect(self.tree_view.open_view)

        splitter.addWidget(board_widget)
        splitter.addWidget(HeuristicViewToggleRadio(self))
        splitter.addWidget(LoadPositionsWidget(self))
        splitter.addWidget(middle_widget)
        splitter.addWidget(noise_widget)
        splitter.addWidget(params_button)
        splitter.addWidget(tree_button)

        splitter.setHandleWidth(10)

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.update_heuristic_parameters()

    def create_search(self, heuristic, player, board):
        return NInARowBestFirstSearch(heuristic, player, board)

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
                self.board = self.board - old_move
                self.move_history.splice_boards([self.board])
                self.on_board_update()

    def show(self):
        """
        Code to display a board as a matplotlib figure
        """
        self.board_renderer.set_board(self.board)

        if self.ghost_check.isChecked() and not self.board.game_has_ended():
            self.board_renderer.add_ghost(
                self.hover, player_to_string(self.board.active_player()))
            self.board_renderer.add_ghost(self.player_ghost, "blue")

        if self.feature_list_toggle.isChecked() and self.feature_list.currentItem():
            for p in pattern_string_to_board_positions(self.feature_list.currentItem().pieces):
                self.board_renderer.add_piece(p, "darkgreen", 0.8)
            for p in pattern_string_to_board_positions(self.feature_list.currentItem().spaces):
                self.board_renderer.add_piece(p, "darkred", 0.8)

        position_values = {}
        if self.heuristic_view:
            for move in self.heuristic_values:
                position_values[(move.get_row(), move.get_col())] = move.val
        else:
            for move in self.candidate_moves:
                position_values[(move.get_move().get_row(),
                                 move.get_move().get_col())] = move.get_value()
        self.board_renderer.set_position_values(position_values)
        self.fig.figure.canvas.draw()

    def reset(self):
        self.clear_history()
        self.set_board(fourbynine_board())

    def clear_history(self):
        self.move_history.clear()

    def set_board(self, board, ghost=None, update_history=True):
        self.board = board
        if update_history:
            self.move_history.set_boards([board])
        self.on_board_update(ghost)

    def play_move(self, new_move):
        if (self.board.contains_spaces(new_move.board_position) and not self.board.game_has_ended()):
            self.board += new_move
            self.move_history.add_board(self.board)
            self.on_board_update()

    def play_best_move(self):
        if not self.board.game_has_ended():
            self.play_move(
                self.heuristic.get_best_move(self.search.get_tree()))

    def on_board_update(self, player_ghost=None):
        self.heuristic_values = self.heuristic.get_moves(
            self.board, self.board.active_player())
        self.hover = None
        self.player_ghost = player_ghost
        self.candidate_moves = []
        self.search = None
        self.search = self.create_search(
            self.heuristic, self.board.active_player(), self.board)
        self.feature_list.update(
            self.heuristic, self.board)

    def dispatch(self):
        self.search.advance_search()
        root = self.search.get_tree()
        if (root):
            self.candidate_moves = root.get_children()

    def update_heuristic_parameters(self):
        self.heuristic = fourbynine_heuristic.create(
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
        super().__init__(board_view)
        self.board_view = board_view
        self.setSortingEnabled(False)
        self.itemDoubleClicked.connect(self._onClick)
        self.itemSelectionChanged.connect(self._itemChanged)
        self.currentItemChanged.connect(self._itemChanged)

    def _onClick(self, item):
        new_move = fourbynine_move(
            item.board_position, 0.0, self.board_view.board.active_player())
        self.board_view.play_move(new_move)

    def _itemChanged(self):
        if self.currentItem():
            self.board_view.hover = self.currentItem().board_position
        else:
            self.board_view.hover = None

    def _update(self, moves):
        old_item = self.currentItem()
        if old_item:
            old_position = old_item.board_position
        else:
            old_position = 0
        self.clear()
        reverse = not player_to_bool(self.board_view.board.active_player())
        for move in sorted(moves, key=fourbynine_game_tree_node.get_value, reverse=reverse):
            self.addItem(MoveListItem(move))
        for i in range(self.count()):
            if (self.item(i).board_position == old_position):
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
        self.pieces = fourbynine_pattern().to_string()
        self.spaces = fourbynine_pattern().to_string()

    def update(self, heuristic, board):
        self.clear()
        feature_group_weights = heuristic.get_feature_group_weights()
        for i, feature in enumerate(heuristic.get_features_with_metadata()):
            feature_group = feature_group_weights[feature.weight_index]
            if feature.feature.contained_in(board, board.active_player()):
                label = "w_act: {}, w_pass: {}, delta: {}".format(
                    feature_group.weight_act, feature_group.weight_pass, feature_group.drop_rate)
                self.addItem(FeatureListItem("idx {}: ".format(
                    i) + label + " " + feature.feature.to_string(), feature.feature))


class MoveHistoryListItem(QListWidgetItem):
    def __init__(self, board):
        super().__init__("{} {}".format(board.get_pieces(Player_Player1).to_string(),
                                        board.get_pieces(Player_Player2).to_string()))
        self.board = board


class MoveHistory(QWidget):
    def __init__(self, board_view):
        super().__init__()

        self.board_view = board_view

        layout = QVBoxLayout()

        self.move_history = QListWidget()
        self.move_history.setSortingEnabled(False)
        self.move_history.itemSelectionChanged.connect(
            self._itemSelectionChanged)
        layout.addWidget(self.move_history)

        self.to_beginning_button = QPushButton("<<")
        self.to_beginning_button.setEnabled(False)
        self.to_beginning_button.clicked.connect(self.to_beginning)
        self.back_button = QPushButton("<")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self.back)
        self.forward_button = QPushButton(">")
        self.forward_button.setEnabled(False)
        self.forward_button.clicked.connect(self.forward)
        self.to_end_button = QPushButton(">>")
        self.to_end_button.setEnabled(False)
        self.to_end_button.clicked.connect(self.to_end)

        navigation_widget = QWidget()
        navigation_layout = QHBoxLayout()
        navigation_widget.setLayout(navigation_layout)

        navigation_layout.addWidget(self.to_beginning_button)
        navigation_layout.addWidget(self.back_button)
        navigation_layout.addWidget(self.forward_button)
        navigation_layout.addWidget(self.to_end_button)

        layout.addWidget(navigation_widget)
        self.setLayout(layout)

        self.clear()

    def to_beginning(self):
        self.set_selection(0)

    def forward(self):
        self.set_selection(self.move_history.currentRow() + 1)

    def back(self):
        self.set_selection(self.move_history.currentRow() - 1)

    def to_end(self):
        self.set_selection(self.move_history.count() - 1)

    def _itemSelectionChanged(self):
        self.board_view.set_board(
            self.move_history.currentItem().board, update_history=False)
        current_idx = self.move_history.currentRow()
        self.to_beginning_button.setEnabled(current_idx != 0)
        self.back_button.setEnabled(current_idx != 0)
        self.forward_button.setEnabled(
            current_idx != self.move_history.count() - 1)
        self.to_end_button.setEnabled(
            current_idx != self.move_history.count() - 1)

    def set_selection(self, idx):
        if (idx >= 0 and idx < self.move_history.count()):
            self.move_history.setCurrentRow(idx)

    def _succeeds(self, second_board, first_board):
        if first_board.num_pieces() + 1 != second_board.num_pieces():
            return False
        last_player = first_board.active_player()
        if first_board.get_pieces(get_other_player(last_player)) != second_board.get_pieces(get_other_player(last_player)):
            return False
        return first_board.missing_pieces(second_board.get_pieces(last_player), last_player).count() == 1

    def splice_boards(self, new_boards):
        if not self.boards or not new_boards:
            self.set_boards(new_boards)
        for i, board in enumerate(self.boards):
            if self._succeeds(new_boards[0], board):
                self.boards = self.boards[:i + 1] + new_boards
                self._update()
                return
        self.set_boards(new_boards)

    def add_board(self, new_board):
        for i, board in enumerate(self.boards):
            if self._succeeds(new_board, board):
                self.boards = self.boards[:i+1]
                self.boards.append(new_board)
                self._update()
                return
        self.set_boards([new_board])

    def clear(self):
        self.boards = [fourbynine_board()]
        self._update()

    def set_boards(self, boards):
        self.boards = boards
        self._update()

    def _update(self):
        self.move_history.clear()
        if not self.boards:
            return
        for board in self.boards:
            self.move_history.addItem(MoveHistoryListItem(board))
        self.to_end()


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


class LoadPositionsWidget(QWidget):
    def __init__(self, board_view):
        super().__init__(board_view)

        self.board_view = board_view
        layout = QVBoxLayout()

        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_widget.setLayout(top_layout)
        layout.addWidget(top_widget)

        top_layout.addWidget(QLabel("Positions:"))
        self.position_combo_box = QComboBox()
        top_layout.addWidget(self.position_combo_box)
        self.display_position_button = QPushButton("Display position")
        self.display_position_button.clicked.connect(self.display_position)
        top_layout.addWidget(self.display_position_button)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        layout.addWidget(bottom_widget)

        bottom_layout.addWidget(QLabel("Detected games:"))
        self.game_combo_box = QComboBox()
        bottom_layout.addWidget(self.game_combo_box)
        self.display_game_button = QPushButton("Display game")
        self.display_game_button.clicked.connect(self.display_game)
        bottom_layout.addWidget(self.display_game_button)

        self.position_combo_box.setEnabled(False)
        self.display_position_button.setEnabled(False)
        self.game_combo_box.setEnabled(False)
        self.display_game_button.setEnabled(False)

        self.load_button = QPushButton("Load positions from file")
        self.load_button.clicked.connect(self.load)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
        self.board_view = board_view

    def parse_move(self, move):
        return (str(move).replace("\t", " "), move.board, move.move)

    def parse_games(self, moves):
        games = []
        i = 0
        while i < len(moves):
            if (moves[i].board == fourbynine_board()):
                j = i
                while (j + 1 < len(moves) and moves[j + 1].board == moves[j].board + moves[j].move):
                    j += 1
                if (i != j):
                    game_name = "{} vs {}: elements {} - {}".format(
                        moves[i].participant_id, moves[i + 1].participant_id, i, j)
                    games.append((game_name, i, j))
                i = j + 1
            else:
                i += 1
        return games

    def load(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            self.position_combo_box.clear()
            moves = []
            for f in filenames:
                try:
                    moves.extend(parse_participant_file(f))
                    for move in moves:
                        display_text, b, m = self.parse_move(move)
                        self.position_combo_box.addItem(display_text, (b, m))
                except Exception:
                    print("Could not parse input file {}".format(f))
            games = self.parse_games(moves)
            for game in games:
                self.game_combo_box.addItem(game[0], (game[1], game[2]))
            positions_exist = self.position_combo_box.count() > 0
            self.position_combo_box.setEnabled(positions_exist)
            self.display_position_button.setEnabled(positions_exist)
            games_exist = self.game_combo_box.count() > 0
            self.game_combo_box.setEnabled(games_exist)
            self.display_game_button.setEnabled(games_exist)
            self.display_position()

    def display_position(self):
        self.set_position(self.position_combo_box.currentIndex())

    def set_position(self, idx):
        if (idx >= 0 and idx < self.position_combo_box.count()):
            self.position_combo_box.setCurrentIndex(idx)
            board, move = self.position_combo_box.itemData(idx)
            self.board_view.clear_history()
            self.board_view.set_board(board, move.board_position)

    def display_game(self):
        idx = self.game_combo_box.currentIndex()
        if idx >= 0:
            bounds = self.game_combo_box.itemData(idx)
            board_list = [self.position_combo_box.itemData(
                x)[0] for x in range(*bounds)]
            self.set_position(bounds[0])
            self.board_view.move_history.set_boards(board_list)


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
        feature_list_layout.addWidget(QLabel("Move history"))
        feature_list_layout.addWidget(self.board.move_history)

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
