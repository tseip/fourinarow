#ifndef SEARCHES_H_INCLUDED
#define SEARCHES_H_INCLUDED

#include <memory>

#include "bfs_node.h"
#include "game_tree_node.h"

/**
 * An abstract search interface, for performing a tree search using a heuristic
 * to evaluate each individual position.
 *
 * @tparam Heuristic The class evaluating each position.
 */
template <class Heuristic>
class AbstractSearch {
 public:
  /**
   * Constructor.
   *
   * @param heuristic The heuristic that will evaluate each position that
   * develops from the board.
   * @param board The board containing the initial position to develop.
   */
  AbstractSearch(std::shared_ptr<Heuristic> heuristic,
                 const typename Heuristic::BoardT &board)
      : heuristic(heuristic), board(board) {
    if (!heuristic)
      throw std::invalid_argument("Must pass a non-null heuristic!");
    this->heuristic->start_search();
  }

  /**
   * Performs a single step of the search algorithm, typically expanding a
   * single node of the search tree (though not necessarily).
   *
   * @return True if the search is complete.
   */
  virtual bool advance_search() = 0;

  /**
   * Runs the current search to completion.
   */
  void complete_search() {
    while (!advance_search()) {
    }
  }

  /**
   * Destructor.
   */
  virtual ~AbstractSearch() {
    if (heuristic) {
      heuristic->complete_search();
    }
  }

  /**
   * @return (the root of) The current search tree.
   */
  virtual std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree() = 0;

 protected:
  /**
   * The heuristic being used to evaluate positions.
   */
  std::shared_ptr<Heuristic> heuristic;

  /**
   * The starting board position.
   */
  const typename Heuristic::BoardT board;
};

/**
 * A concrete search interface, for performing a tree search using a heuristic
 * to evaluate each individual position, which manages the nodes of the search
 * tree.
 *
 * @tparam Heuristic The class evaluating each position.
 * @tparam NodeT The class representing a single node in the search tree.
 */
template <class Heuristic, class NodeT>
class Search : public AbstractSearch<Heuristic> {
 public:
  /**
   * Constructor.
   *
   * @param heuristic The heuristic that will evaluate each position that
   * develops from the board.
   * @param board The board containing the initial position to develop.
   */
  Search(std::shared_ptr<Heuristic> heuristic,
         const typename Heuristic::BoardT &board)
      : AbstractSearch<Heuristic>(heuristic, board),
        root(NodeT::create(board, heuristic->evaluate(board))) {}

  /**
   * Performs a single step of the search algorithm, typically expanding a
   * single node of the search tree (though not necessarily).
   *
   * @return True if the search is complete.
   */
  virtual bool advance_search() override {
    if (stopping_conditions(this->heuristic, this->board)) {
      this->heuristic->complete_search();
      return true;
    } else {
      auto current_node = select_next_node();
      const auto current_board = current_node->get_board();
      const std::vector<typename Heuristic::BoardT::MoveT> candidate_moves =
          this->heuristic->get_pruned_moves(current_board,
                                            current_board.active_player());
      current_node->expand(candidate_moves);
      on_node_expansion(current_node, this->heuristic, this->board);
      return false;
    }
  }

  /**
   * @return (the root of) The current search tree.
   */
  virtual std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree()
      override {
    return std::dynamic_pointer_cast<Node<typename Heuristic::BoardT>>(root);
  }

 protected:
  /**
   * Finds the next node in the tree to be expanded at each step of the search
   * process.
   *
   * @return The next node to be expanded/evaluated.
   */
  virtual std::shared_ptr<NodeT> select_next_node() {
    return std::dynamic_pointer_cast<NodeT>(root->select());
  }

  /**
   * Evaluates search stopping conditions and returns true if the search should
   * end.
   *
   * @param heuristic The heuristic that will evaluate each position that
   * develops from the board.
   * @param board The board containing the initial position to develop.
   *
   * @return True if the search should stop, false otherwise.
   */
  virtual bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic,
      const typename Heuristic::BoardT &board) const {
    return this->root->determined();
  }

  /**
   * Called whenever a node is expanded. Can be overridden to enable tracking of
   * search metadata.
   *
   * @param expanded_node The node that was just expanded.
   * @param heuristic The heuristic that will evaluate each position that
   * develops from the board.
   * @param board The board containing the initial position to develop.
   */
  virtual void on_node_expansion(std::shared_ptr<NodeT> expanded_node,
                                 std::shared_ptr<Heuristic> heuristic,
                                 const typename Heuristic::BoardT &board) {}

  /**
   * The root of the search tree.
   */
  std::shared_ptr<NodeT> root;
};

#endif  // SEARCHES_H_INCLUDED
