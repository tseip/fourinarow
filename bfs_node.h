#ifndef BFS_NODE_H_INCLUDED
#define BFS_NODE_H_INCLUDED

#include <iostream>
#include <limits>
#include <memory>

#include "game_tree_node.h"
#include "player.h"

/**
 * Represents a single node in the game tree.
 *
 * @tparam Board The board representation used by the game.
 */
template <class Board>
class BFSNode : public Node<Board> {
 private:
  using Node<Board>::children;
  using Node<Board>::board;
  using Node<Board>::depth;
  using Node<Board>::parent;
  using Node<Board>::Node;
  using Node<Board>::update_field_against_child;

  static std::shared_ptr<BFSNode<Board>> downcast(
      std::shared_ptr<Node<Board>> node) {
    return std::dynamic_pointer_cast<BFSNode<Board>>(node);
  }

  static std::shared_ptr<const BFSNode<Board>> downcast(
      std::shared_ptr<const Node<Board>> node) {
    return std::dynamic_pointer_cast<const BFSNode<Board>>(node);
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
   * The child of this node with the best known heuristic value.
   */
  std::shared_ptr<BFSNode> best_known_child;

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
   * Creates a node on the heap that is a child of the current node and returns
   * a pointer to it.
   *
   * @param move The move to be represented by the child.
   *
   * @return A pointer to the newly created node.
   */
  std::shared_ptr<BFSNode> create_child(const typename Board::MoveT &move) {
    // Validate that the parent doesn't already have this child.
    for (const auto &child : children) {
      if (downcast(child)->move.board_position == move.board_position) {
        throw std::logic_error(
            "Given move already exists as a child of this node!");
      }
    }

    return std::shared_ptr<BFSNode>(
        new BFSNode(downcast(this->shared_from_this()), move));
  }

  /**
   * Private node constructor for nodes without meaningful move history (i.e.,
   * with no parents).
   *
   * @param board The board state prior to the move being represented by this
   * node is made.
   * @param val The heuristic value of the move this node represents.
   */
  BFSNode(const Board &board, double val)
      : Node<Board>(board), val(val), pess(0), opt(0) {
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
  BFSNode(const std::shared_ptr<BFSNode> parent,
          const typename Board::MoveT &move)
      : Node<Board>(parent, move),
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
   * Updates the opt field of this node by scanning over all of its children.
   * Assumes all of this node's children are themselves updated.
   */
  void update_opt() {
    opt = (board.active_player() == Player::Player1 ? WHITE_WINS + depth
                                                    : BLACK_WINS - depth);
    for (const auto &child : children) {
      update_field_against_child(downcast(child)->opt, opt);
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
      update_field_against_child(downcast(child)->pess, pess);
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
      if (!downcast(child)->determined() &&
          update_field_against_child(downcast(child)->val, val)) {
        best_known_child = downcast(child);
      }
    }

    for (const auto &child : children) {
      if (downcast(child)->determined()) {
        update_field_against_child(downcast(child)->val, val);
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
  void backpropagate(const std::shared_ptr<BFSNode> &child) {
    if (!update_field_against_child(child->opt, opt)) update_opt();
    if (!update_field_against_child(child->pess, pess)) update_pess();
    if (!child->determined() && update_field_against_child(child->val, val)) {
      best_known_child = child;
    } else {
      update_val();
      update_best_determined();
    }

    if (parent)
      downcast(parent)->backpropagate(downcast(this->shared_from_this()));
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
        if (downcast(child)->pess == pess) {
          best_known_child = downcast(child);
          break;
        }
      }
    } else {
      for (const auto &child : children) {
        if (downcast(child)->opt == opt) {
          best_known_child = downcast(child);
          break;
        }
      }
    }
  }

 public:
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
  static std::shared_ptr<BFSNode> create(const Board &board, double val) {
    return std::shared_ptr<BFSNode>(new BFSNode(board, val));
  }

  /**
   * @return True if the heuristic value of this node has converged, i.e. if the
   * pessimistic and optimistic bounds on the value of the node are equal.
   */
  bool determined() const { return pess == opt; }

  /**
   * @return A string representing the state of this node.
   */
  std::string to_string() const override {
    std::stringstream stream;
    stream << Node<Board>::to_string() << ", Heuristic value: " << val
           << ", Opt: " << opt << ", Pess: " << pess;
    return stream.str();
  }

  /**
   * Accepts a list of moves that can be played from this position represented
   * by this node and adds them to the game tree.
   *
   * @param moves A list of moves that can be played from this position.
   */
  void expand(const std::vector<typename Board::MoveT> &moves) override {
    if (moves.empty()) return;

    for (const typename Board::MoveT &move : moves) {
      children.push_back(create_child(move));
    }

    update_opt();
    update_pess();
    update_val();
    if (determined()) update_best_determined();
    if (parent)
      downcast(parent)->backpropagate(downcast(this->shared_from_this()));
  }

  /**
   * Select the best move from among our children. If we have no good child,
   * return ourselves. Note that the returned move may be multiple moves in the
   * future, i.e. we may return a move that represents an entire line of play.
   *
   * @return The best move from amongst our children and ourselves.
   */
  std::shared_ptr<const Node<Board>> select() const override {
    if (best_known_child) {
      return best_known_child->select();
    } else {
      return downcast(this->shared_from_this());
    }
  }

  /**
   * @return The number of moves between us and our recursively best known
   * child.
   */
  std::size_t get_depth_of_pv() const {
    const auto selected_node = select();
    if (selected_node == this->shared_from_this()) return 0;
    return downcast(selected_node)->depth - depth - 1;
  }

  /**
   * @return The best known move from the current position for the current
   * player.
   */
  typename Board::MoveT get_best_move() const {
    if (!best_known_child)
      throw std::logic_error(
          "No best known child has been determined for this board:\n" +
          board.to_string());
    if (determined()) {
      return typename Board::MoveT(best_known_child->move.board_position, val,
                                   board.active_player());
    }
    double val_temp = (board.active_player() == Player::Player1
                           ? -std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::infinity());

    if (children.empty()) return this->move;

    std::size_t best_position =
        downcast(this->children[0])->move.board_position;
    for (const auto &child : children) {
      if (update_field_against_child(downcast(child)->val, val_temp)) {
        best_position = downcast(child)->move.board_position;
      }
    }

    return
        typename Board::MoveT(best_position, val_temp, board.active_player());
  }
};

#endif  // BFS_NODE_H_INCLUDED
