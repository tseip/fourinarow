from fourbynine import *
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import networkx as nx
import random


def pattern_string_to_board_positions(pattern_string):
    """
    Converts a given pattern string (gotten from calling pattern.to_string())
    to a list of board positions (numbered from 0 to 35 from the upper left)
    corresponding to set bits in the pattern.

    Args:
        pattern_string: The pattern to convert.

    Returns:
        A list of 0-indexed positions corresponding to the bits set in the pattern.
    """
    output = []
    for i in range(len(pattern_string)):
        if pattern_string[-i-1] == "1":
            output.append(i)
    return output


class BoardRenderer():
    """
    Renders a n-in-a-row board using matplotlib.
    """

    def __init__(self, ax):
        """
        Constructor.

        Params:
            ax: The matplotlib axes to render the board onto.
        """
        self.ax = ax

        self.clear()

    def add_piece(self, position, color, alpha=1.0):
        """
        Adds a piece to the board.

        Args:
            position: The 0-indexed position to add the piece to.
            color: The color of the piece.
            alpha: The alpha of the piece.
        """
        if position is not None:
            piece = patches.Circle((position % fourbynine_board().get_board_width(), position//fourbynine_board().get_board_width()), 0.33,
                                   color=color, fill=True, alpha=alpha)
            self.ax.add_patch(piece)
            self.pieces.add(position)

    def add_ghost(self, position, color):
        """
        Adds a transparent piece at the given position if there isn't already a piece ther.

        Args:
            position: The position to add the ghost to.
            color: The color of the ghost.
        """
        if position not in self.pieces:
            self.add_piece(position, color, 0.5)

    def set_board(self, board):
        """
        Sets the board to the given board.

        Args:
            board: The board to set the display to.
        """
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
        """
        Given an array of heuristic values, color the spaces on the board to correspond to the values.
        White/black correspond to a guaranteed win for the given player, shades of green/red correspond
        to "good" or "bad" moves for the current player. Gray corresponds to a draw.

        Args:
            position_values: The values of each space as determined by a heuristic.
        """
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
        """
        Clears the board.
        """
        self.ax.clear()
        self.ax.axis('off')
        self.board = None
        self.pieces = set()


class SearchRenderer():
    """
    Renders a search tree in matplotlib.
    """

    def __init__(self, ax):
        """
        Constructor.

        Params:
            ax: The matplotlib axes to render the board onto.
        """
        self.ax = ax
        self.root = None
        self.g = None
        self.max_branching_factor = 5
        self.max_depth = 0
        self.board_size = 0.3
        self.board_axes = []
        self.onclick_callback = lambda board: None
        self.clear()

    def _clear_board_axes(self):
        for axis, board in self.board_axes:
            axis.remove()
        self.board_axes.clear()

    def register_onclick_callback(self, cb):
        """
        Registers a callback to be called when an element in the
        search is clicked. The callback will be given a list of boards
        that have been clicked.

        Args:
            cb: The callback to be called.
        """
        self.onclick_callback = cb

    def clear(self):
        """
        Clears the display.
        """
        self._clear_board_axes()
        self.ax.clear()
        self.ax.axis('off')
        self.root = None
        self.g = None

    def set_root(self, root):
        """
        Populates the display using the given root of the search tree.

        Args:
            root: The root of the search tree.
        """
        self.clear()
        if not root:
            return
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
        reverse = not player_to_bool(self.root.get_board().active_player())
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
            for i, child in enumerate(sorted(node.get_children(), key=fourbynine_game_tree_node.get_value, reverse=reverse)):
                if self.max_branching_factor and i >= self.max_branching_factor:
                    break
                nodes_to_process.append(child)

    def draw(self):
        """
        Draws the search tree.
        """
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
        """
        Should be connected to an on-click event on the renderer's canvas. Calls the callback
        in the event of a double click with the board that was clicked.

        Args:
            event: The click event that has fired.
        """
        if event.dblclick and event.button == 1:
            for axis, board in self.board_axes:
                if (axis.in_axes(event)):
                    path = nx.shortest_path(
                        self.g, source=self.root.get_board().to_string(), target=board.to_string())
                    attributes = nx.get_node_attributes(self.g, "board")
                    boards = [attributes[n] for n in path]
                    self.onclick_callback(boards)


def main():
    fig, ax = plt.subplots()
    board = fourbynine_board(fourbynine_pattern(0), fourbynine_pattern(0))
    heuristic = fourbynine_heuristic.create()
    heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(False)
    bfs = NInARowBestFirstSearch(heuristic, board)
    bfs.complete_search()
    root = bfs.get_tree()
    renderer = SearchRenderer(ax)
    renderer.register_onclick_callback(
        lambda boards: print(boards[-1].to_string()))
    fig.canvas.mpl_connect('button_press_event', renderer.onclick)
    renderer.set_root(root)
    renderer.draw()
    plt.show()


if __name__ == "__main__":
    main()
