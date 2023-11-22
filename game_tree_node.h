#ifndef GAME_TREE_NODE_H_INCLUDED
#define GAME_TREE_NODE_H_INCLUDED

#include <memory>
#include <queue>
#include <sstream>

#include "player.h"

/**
 * Represents a single node in the game tree.
 *
 * @tparam Board The board representation used by the game.
 */
template <class Board>
class Node : public std::enable_shared_from_this<Node<Board>> {
 public:
  /**
   * Allows for BFS iteration over the game tree.
   */
  template <class It>
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = It;
    using pointer = std::shared_ptr<value_type>;
    using reference = value_type &;

    /**
     * Constructor.
     *
     * @param ptr The root node of the tree to iterate over.
     */
    Iterator(pointer ptr) : nodes() {
      if (ptr) nodes.push(ptr);
    }

    /**
     * @return The node the iterator points to.
     */
    reference operator*() const { return *nodes.front(); }

    /**
     * @return The node the iterator points to.
     */
    pointer operator->() { return nodes.front(); }

    /**
     * Prefix increment.
     *
     * @return The next node in the iterator.
     */
    Iterator &operator++() {
      for (auto &node : (*this)->get_children()) {
        nodes.push(node);
      }
      nodes.pop();
      return *this;
    }

    /**
     * Prefix increment.
     *
     * @return The node the iterator points to.
     */
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    /**
     * Iterator comparison functions.
     *
     * @param a The first iterator to compare.
     * @param b The second iterator to compare.
     *
     * @{
     */
    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.nodes == b.nodes;
    };
    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.nodes != b.nodes;
    };
    /**
     * @}
     */

   private:
    /**
     * The queue of nodes yet to be iterated over.
     */
    std::queue<pointer> nodes;
  };

  /**
   * @return An iterator that will iterate over the game tree under this node.
   */
  Iterator<Node> begin() { return Iterator<Node>(this->shared_from_this()); }

  /**
   * @return An empty iterator.
   */
  Iterator<Node> end() { return Iterator<Node>(nullptr); }

  /**
   * @return An iterator that will iterate over the game tree under this node.
   */
  Iterator<const Node> begin() const {
    return Iterator<const Node>(this->shared_from_this());
  }

  /**
   * @return An empty iterator.
   */
  Iterator<const Node> end() const { return Iterator<const Node>(nullptr); }

  /**
   * @return The node's heuristic value, determined by the search algorithm
   * used.
   * @note Since the base class has no search algorithm associated with it,
   * we order arbitrarily by board position.
   */
  virtual double get_value() const { return move.board_position; }

  /**
   * @return True if this node is solved, i.e. if the outcome of the game is
   * known from this position.
   */
  virtual bool determined() const = 0;

  bool operator<(const Node &other) { return get_value() < other.get_value(); }

  /**
   * @return The board state represented by this node.
   */
  const Board get_board() const { return board; }

  /**
   * @return The move prior to the board state represented by this node.
   */
  const typename Board::MoveT get_move() const { return move; }

  /**
   * @return The children of this node.
   */
  std::vector<std::shared_ptr<Node>> &get_children() { return children; }

  /**
   * @return The children of this node.
   */
  const std::vector<std::shared_ptr<Node>> &get_children() const {
    return children;
  }

  /**
   * @return The depth of this node in the tree.
   */
  std::size_t get_depth() const { return depth; }

  /**
   * @return The parent of this node in the tree.
   */
  std::shared_ptr<Node> get_parent() const { return parent.lock(); }

  /**
   * @return A string representing the state of this node.
   */
  virtual std::string to_string() const {
    std::stringstream stream;
    stream << "Position: " << move.board_position
           << ", Player: " << static_cast<size_t>(move.player)
           << ", Depth: " << depth;
    return stream.str();
  }

  /**
   * @param How many layers of children should also be printed.
   *
   * @return A string representing the state of this node.
   */
  std::string to_string(std::size_t max_depth) const {
    std::stringstream stream;
    for (auto &node : *this) {
      if (node.depth - depth >= max_depth) break;
      stream << node.to_string() << std::endl;
    }
    return stream.str();
  }

 protected:
  /**
   * The children of this node.
   */
  std::vector<std::shared_ptr<Node>> children;

  /**
   * The parent of this node.
   */
  const std::weak_ptr<Node> parent;

  /**
   * The depth of this node in the game tree.
   */
  const std::size_t depth;

  /**
   * The board state of this node with the move that this node represents
   * included.
   */
  const Board board;

  /**
   * The move that led to the current board state.
   */
  const typename Board::MoveT move;

  /**
   * Private node constructor for nodes without meaningful move history (i.e.,
   * with no parents).
   *
   * @param board The board state prior to the move being represented by this
   * node is made.
   */
  Node(const Board &board)
      : children(), parent(), depth(1U), board(board), move() {}

  /**
   * Private node constructor for nodes with meaningful move history (i.e., with
   * parents).
   *
   * @note Checking for nullity of parent is presumed to have been done outside
   * of this function.
   *
   * @param parent The parent of this node.
   * @param move The move represented by this node.
   */
  Node(const std::shared_ptr<Node> parent, const typename Board::MoveT &move)
      : children(),
        parent(parent),
        depth(1U + parent->depth),
        board(parent->board + move),
        move(move) {}

  /**
   * Helper function for updating the val, opt, and pess fields while respecting
   * each player's outcome sign preference (Player1 prefers positive values,
   * Player2 prefers negative values).
   *
   * @tparam Field The type of the field to be updated.
   * @param child_field The field to compare against.
   * @param field The field being compared and updated if the child_field is
   * preferential for the player of this node's move.
   *
   * @return True if the field has been updated, false otherwise.
   */
  template <class Field>
  bool update_field_against_child(const Field &child_field,
                                  Field &field) const {
    if (board.active_player() == Player::Player1) {
      if (child_field > field) {
        field = child_field;
        return true;
      }
    } else {
      if (child_field < field) {
        field = child_field;
        return true;
      }
    }

    return false;
  }

  /**
   * Sums the depth of all of the leaf nodes in the game tree beneath us.
   *
   * @return The sum of the depth of all of the leaf nodes in the tree beneath
   * us, including us.
   */
  std::size_t get_sum_depth() const {
    if (children.empty()) {
      return depth;
    }
    std::size_t n = 0;
    for (const auto &child : children) {
      n += child->get_sum_depth();
    }
    return n;
  }

 public:
  /**
   * Accepts a list of moves that can be played from this position represented
   * by this node and adds them to the game tree.
   *
   * @param moves A list of moves that can be played from this position.
   */
  virtual void expand(const std::vector<typename Board::MoveT> &moves) = 0;

  /**
   * @return The number of moves between us and our recursively best known
   * child.
   */
  virtual std::size_t get_depth_of_pv() const = 0;

  /**
   * Select the next move to be searched from among our children recursively.
   *
   * @note We delegate this to a virtual function so that we don't have to
   * define a const and non-const version of select in all subclasses.
   *
   * @return The best move from amongst our children and ourselves.
   */
  std::shared_ptr<const Node> select() const { return virtual_select(); }

  /**
   * Select the next move to be searched from among our children recursively.
   *
   * @return The next move from amongst our children and ourselves that ought to
   * be expanded.
   */
  std::shared_ptr<Node> select() {
    return std::const_pointer_cast<Node>(
        std::const_pointer_cast<const Node>(this->shared_from_this())
            ->select());
  }

  /**
   * Find the number of leaf nodes beneath us, including us. A leaf node is
   * defined as a node that has no best known children.
   *
   * @return The number of leaf nodes beneath us, including us.
   */
  std::size_t get_num_leaves() const {
    if (children.empty()) {
      return 1;
    }
    std::size_t n = 0;
    for (const auto &child : children) {
      n += child->get_num_leaves();
    }
    return n;
  }

  /**
   * @return The number of all nodes in the tree.
   */
  std::size_t get_node_count() const {
    std::size_t node_count = 1;
    for (const auto &child : children) {
      node_count += child->get_node_count();
    }
    return node_count;
  }

  /**
   * @return The average branching factor of the tree.
   */
  double get_average_branching_factor() const {
    if (children.empty()) return 0.0;
    return static_cast<double>(get_node_count() - 1) / get_num_internal_nodes();
  }

  /**
   * Find the number of internal nodes beneath us. Only nodes with
   * children are internal nodes.
   *
   * @return The number of internal nodes beneath us, including us.
   */
  std::size_t get_num_internal_nodes() const {
    if (children.empty()) {
      return 0;
    }
    std::size_t n = 1;
    for (const auto &child : children) {
      n += child->get_num_internal_nodes();
    }
    return n;
  }

  /**
   * @return The average depth of all of the leaf nodes in the tree beneath us,
   * including us.
   */
  double get_mean_depth() const {
    return (static_cast<double>(get_sum_depth()) / get_num_leaves());
  }

  /**
   * @return The best known move from the current position for the current
   * player.
   */
  virtual typename Board::MoveT get_best_move() const = 0;

 protected:
  /**
   * Select the next move to be searched from among our children recursively.
   *
   * @note We split this out so that we don't have to define a const and
   * non-const version of select in all subclasses.
   *
   * @return The best move from amongst our children and ourselves.
   */
  virtual std::shared_ptr<const Node> virtual_select() const = 0;
};

#endif  // GAME_TREE_NODE_H_INCLUDED
