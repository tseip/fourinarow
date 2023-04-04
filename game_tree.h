#ifndef GAME_TREE_H_INCLUDED
#define GAME_TREE_H_INCLUDED

#include <memory>
#include <queue>

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
   * Creates a node with no move history on the heap and returns a pointer to
   * it.
   *
   * @param board The board state prior to the move being represented by this
   * node is made.
   * @param val The heuristic value of the move this node represents.
   *
   * @return A pointer to the newly created node.
   */
  static std::shared_ptr<Node> create(const Board &board, double val) {
    return std::shared_ptr<Node>(new Node(board, val));
  }

  /**
   * @return True if the heuristic value of this node has converged, i.e. if the
   * pessimistic and optimistic bounds on the value of the node are equal.
   */
  bool determined() const { return pess == opt; }

  /**
   * @return The board state represented by this node.
   */
  const Board get_board() const { return board; }

  /**
   * @return The move prior to the board state represented by this node.
   */
  const typename Board::Move get_move() const { return move; }

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
  std::shared_ptr<Node> get_parent() const { return parent; }

  /**
   * @return A string representing the state of this node.
   */
  std::string to_string() const {
    std::stringstream stream;
    stream << "Position: " << move.board_position
           << ", Player: " << static_cast<size_t>(move.player)
           << ", Depth: " << depth << ", Heuristic value: " << val
           << ", Opt: " << opt << ", Pess: " << pess << std::endl;
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

 private:
  /**
   * Creates a node on the heap that is a child of the current node and returns
   * a pointer to it.
   *
   * @param move The move to be represented by the child.
   *
   * @return A pointer to the newly created node.
   */
  std::shared_ptr<Node> create_child(const typename Board::Move &move) {
    // Validate that the parent doesn't already have this child.
    for (const auto &child : children) {
      if (child->move.board_position == move.board_position) {
        throw std::logic_error(
            "Given move already exists as a child of this node!");
      }
    }

    return std::shared_ptr<Node>(new Node(this->shared_from_this(), move));
  }

  /**
   * Heuristic values for a black win and white win, respectively. This is the
   * maximum tree depth, which is the maximum number of moves plus one, plus
   * one.
   * @{
   */
  static constexpr int BLACK_WINS = Board::get_max_num_moves() + 1 + 1;
  static constexpr int WHITE_WINS = -BLACK_WINS;
  /**
   * @}
   */

  /**
   * The children of this node.
   */
  std::vector<std::shared_ptr<Node>> children;

  /**
   * The parent of this node.
   */
  const std::shared_ptr<Node> parent;

  /**
   * The child of this node with the best known heuristic value.
   */
  std::shared_ptr<Node> best_known_child;

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
  const typename Board::Move move;

  /**
   * The heuristic value of this node.
   */
  double val;

  /**
   * A pessimistic estimate of the true expected value of this node.
   */
  int pess;

  /**
   * An optimistic estimate of the true expected value of this node.
   */
  int opt;

  /**
   * Private node constructor for nodes without meaningful move history (i.e.,
   * with no parents).
   *
   * @param board The board state prior to the move being represented by this
   * node is made.
   * @param val The heuristic value of the move this node represents.
   */
  Node(const Board &board, double val)
      : children(),
        parent(nullptr),
        best_known_child(),
        depth(1U),
        board(board),
        move(),
        val(val),
        pess(0),
        opt(0) {
    setup_pess_opt();
  }

  /**
   * Private node constructor for nodes with meaningful move history (i.e., with
   * parents).
   *
   * @note Checking for nullity of parent is presumed to have been done outside
   * of this function.
   *
   * @param board The board state prior to the move being represented by this
   * node is made.
   * @param move The move represented by this node.
   * @param val The heuristic value of the move this node represents.
   * @param parent The parent of this node, if any.
   */
  Node(const std::shared_ptr<Node> parent, const typename Board::Move &move)
      : children(),
        parent(parent),
        best_known_child(),
        depth(1U + parent->depth),
        board(parent->board + move),
        move(move),
        val(parent->board.active_player() == Player::Player1
                ? parent->val + move.val
                : parent->val - move.val),
        pess(0),
        opt(0) {
    setup_pess_opt();
  }

  /**
   * Establishes initial values for pess and opt based on the board state.
   */
  void setup_pess_opt() {
    if (board.player_has_won(Player::Player1))
      pess = opt = BLACK_WINS - depth,
      val = std::numeric_limits<double>::infinity();
    else if (board.player_has_won(Player::Player2))
      pess = opt = WHITE_WINS + depth,
      val = -std::numeric_limits<double>::infinity();
    else if (board.game_is_drawn())
      pess = opt = 0, val = 0.0;
    else
      pess = WHITE_WINS + depth, opt = BLACK_WINS - depth;
  }

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
   * Updates the opt field of this node by scanning over all of its children.
   * Assumes all of this node's children are themselves updated.
   */
  void update_opt() {
    opt = (board.active_player() == Player::Player1 ? WHITE_WINS + depth
                                                    : BLACK_WINS - depth);
    for (const auto &child : children) {
      update_field_against_child(child->opt, opt);
    }
  }

  /**
   * Updates the pess field of this node by scanning over all of its children.
   * Assumes all of this node's children are themselves updated.
   */
  void update_pess() {
    pess = (board.active_player() == Player::Player1 ? WHITE_WINS + depth
                                                     : BLACK_WINS - depth);
    for (const auto &child : children) {
      update_field_against_child(child->pess, pess);
    }
  }

  /**
   * Updates the val field of this node by scanning over all of its children.
   * Assumes all of this node's children are themselves updated.
   */
  void update_val() {
    val = (board.active_player() == Player::Player1
               ? -std::numeric_limits<double>::infinity()
               : std::numeric_limits<double>::infinity());
    for (const auto &child : children) {
      if (!child->determined() && update_field_against_child(child->val, val)) {
        best_known_child = child;
      }
    }

    for (const auto &child : children) {
      if (child->determined()) {
        update_field_against_child(child->val, val);
      }
    }
  }

  /**
   * Takes a (presumably) updated child of this node, updates the current node
   * with any new game state information in the child, and then reports that
   * information recursively to the parent of this node up to the root of the
   * game tree.
   *
   * @param child The child node that has been updated and whose new values
   * should be propagated to the root of the tree.
   */
  void backpropagate(const std::shared_ptr<Node> &child) {
    if (!update_field_against_child(child->opt, opt)) update_opt();
    if (!update_field_against_child(child->pess, pess)) update_pess();
    if (!child->determined() && update_field_against_child(child->val, val)) {
      best_known_child = child;
    } else {
      update_val();
      update_best_determined();
    }

    if (parent) parent->backpropagate(this->shared_from_this());
  }

  /**
   * Scans through the children of this node to find the first child whose
   * pessimistic or optimistic estimate matches our own, and chooses that to be
   * the best known child. This is only valid if we ourselves are determined
   * (i.e., if our optimistic/pessimistic estimates for the heuristic value of
   * this node have converged).
   */
  void update_best_determined() {
    // Nothing to do if we aren't ourselves determined.
    if (!determined()) {
      return;
    }

    if (board.active_player() == Player::Player1) {
      for (const auto &child : children) {
        if (child->pess == pess) {
          best_known_child = child;
          break;
        }
      }
    } else {
      for (const auto &child : children) {
        if (child->opt == opt) {
          best_known_child = child;
          break;
        }
      }
    }
  }

  /**
   * Sums the depth of all of the leaf nodes in the game tree beneath us.
   *
   * @return The sum of the depth of all of the leaf nodes in the tree beneath
   * us, including us.
   */
  std::size_t get_sum_depth() const {
    if (!best_known_child) {
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
  void expand(const std::vector<typename Board::Move> &moves) {
    if (moves.empty()) return;

    for (const typename Board::Move &move : moves) {
      children.push_back(create_child(move));
    }

    update_opt();
    update_pess();
    update_val();
    if (determined()) update_best_determined();
    if (parent) parent->backpropagate(this->shared_from_this());
  }

  /**
   * Select the best move from among our children. If we have no good child,
   * return ourselves. Note that the returned move may be multiple moves in the
   * future, i.e. we may return a move that represents an entire line of play.
   *
   * @return The best move from amongst our children and ourselves.
   */
  std::shared_ptr<const Node> select() const {
    if (best_known_child) return best_known_child->select();
    return this->shared_from_this();
  }

  /**
   * Select the best move from among our children. If we have no good child,
   * return ourselves. Note that the returned move may be multiple moves in the
   * future, i.e. we may return a move that represents an entire line of play.
   *
   * @return The best move from amongst our children and ourselves.
   */
  std::shared_ptr<Node> select() {
    return std::const_pointer_cast<Node>(
        const_cast<const Node *>(this)->select());
  }

  /**
   * Find the number of leaf nodes beneath us, including us. A leaf node is
   * defined as a node that has no best known children.
   *
   * @return The number of leaf nodes beneath us, including us.
   */
  std::size_t get_num_leaves() const {
    if (!best_known_child) {
      return 1;
    }
    std::size_t n = 0;
    for (const auto &child : children) {
      n += child->get_num_leaves();
    }
    return n;
  }

  /**
   * Find the number of internal nodes beneath us. Only nodes with best known
   * children are internal nodes.
   *
   * @return The number of internal nodes beneath us, including us.
   */
  std::size_t get_num_internal_nodes() const {
    if (!best_known_child) {
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
   * @return The number of moves between us and our recursively best known
   * child.
   */
  std::size_t get_depth_of_pv() const {
    if (!best_known_child) return 0;
    return select()->depth - depth - 1;
  }

  /**
   * @return The best known move from the current position for the current
   * player.
   */
  typename Board::Move get_best_move() const {
    if (!best_known_child)
      throw std::logic_error(
          "No best known child has been determined for this board:\n" +
          board.to_string());
    if (determined()) {
      return typename Board::Move(best_known_child->move.board_position, val,
                                  board.active_player());
    }
    double val_temp = (board.active_player() == Player::Player1
                           ? -std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::infinity());
    std::size_t best_position;
    for (const auto &child : children) {
      if (update_field_against_child(child->val, val_temp)) {
        best_position = child->move.board_position;
      }
    }

    return typename Board::Move(best_position, val_temp, board.active_player());
  }
};

#endif  // GAME_TREE_H_INCLUDED
