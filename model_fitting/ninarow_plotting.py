from fourbynine import *
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import networkx as nx
import random


def pattern_string_to_board_positions(pattern_string):
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output


class BoardRenderer():
    def __init__(self, ax):
        self.ax = ax

        self.clear()

    def add_piece(self, position, color, alpha=1.0):
        if position is not None:
            piece = patches.Circle((position % fourbynine_board().get_board_width(), position//fourbynine_board().get_board_width()), 0.33,
                                   color=color, fill=True, alpha=alpha)
            self.ax.add_patch(piece)
            self.pieces.add(position)

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

    def clear(self):
        self.ax.clear()
        self.ax.axis('off')
        self.board = None
        self.pieces = set()


class SearchRenderer():
    def __init__(self, ax):
        self.ax = ax
        self.root = None
        self.g = None
        self.max_branching_factor = 5
        self.max_depth = None
        self.board_size = 0.3
        self.board_axes = []
        self.onclick_callback = lambda board: None
        self.clear()

    def _clear_board_axes(self):
        for axis, board in self.board_axes:
            axis.remove()
        self.board_axes.clear()

    def register_onclick_callback(self, cb):
        self.onclick_callback = cb

    def clear(self):
        self._clear_board_axes()
        self.ax.clear()
        self.ax.axis('off')
        self.root = None
        self.g = None

    def set_max_branching_factor(self, max_branching_factor):
        self.max_branching_factor = max_branching_factor
        self.set_root(self.root)

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth
        self.set_root(self.root)

    def set_board_size(self, board_size):
        self.board_size = board_size
        self.draw()

    def set_root(self, root):
        self.clear()
        self.root = root
        self._populate_graph()
        self.pos = nx.multipartite_layout(self.g, scale=1)
        for node, pos in self.pos.items():
            self.pos[node][1] = self.pos[node][1]*self.max_branching_factor
            self.pos[node][0] = self.pos[node][0]
        nx.draw_networkx_edges(self.g, pos=self.pos, ax=self.ax)
        nx.draw_networkx_edge_labels(self.g, pos=self.pos, ax=self.ax, edge_labels={
                                     edge: self.g.get_edge_data(*edge)["weight"] for edge in self.g.edges()})
        self.draw()

    def _populate_graph(self):
        self.g = nx.Graph()
        active_player = self.root.get_board().active_player()
        nodes_to_process = [self.root]
        while nodes_to_process:
            node = nodes_to_process.pop(0)
            if self.max_depth and node.get_depth() > self.max_depth:
                break
            self.g.add_node(node.get_board().to_string(),
                            subset=node.get_depth(), board=node.get_board())
            if (node.get_parent()):
                self.g.add_edge(node.get_parent().get_board().to_string(
                ), node.get_board().to_string(), weight=node.get_value())
            for i, child in enumerate(sorted(node.get_children(), key=fourbynine_game_tree_node.get_value, reverse=active_player == node.get_board().active_player())):
                if self.max_branching_factor and i >= self.max_branching_factor:
                    break
                nodes_to_process.append(child)

    def draw(self):
        if not self.g:
            return

        self._clear_board_axes()
        board_center = self.board_size / 2.0

        # Add the respective image to each node
        for n in self.g.nodes:
            new_axis = self.ax.inset_axes([self.pos[n][0] - board_center, self.pos[n][1] -
                                          board_center, self.board_size, self.board_size], transform=self.ax.transData)
            board = self.g.nodes[n]["board"]
            self.board_axes.append((new_axis, board))
            br = BoardRenderer(new_axis)
            br.set_board(board)

    def onclick(self, event):
        for axis, board in self.board_axes:
            if (axis.in_axes(event)):
                self.onclick_callback(board)


def main():
    fig, ax = plt.subplots()
    board = fourbynine_board(fourbynine_pattern(0), fourbynine_pattern(0))
    heuristic = fourbynine_heuristic.create()
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(False)
    bfs = NInARowBestFirstSearch.create()
    bfs.search(heuristic, board.active_player(), board)
    root = bfs.get_tree()
    renderer = SearchRenderer(ax)
    renderer.register_onclick_callback(lambda board: print(board.to_string()))
    fig.canvas.mpl_connect('button_press_event', renderer.onclick)
    renderer.set_root(root)
    renderer.draw()
    plt.show()


if __name__ == "__main__":
    main()
