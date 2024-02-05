#ifndef SEARCHES_H_INCLUDED
#define SEARCHES_H_INCLUDED

#include <memory>

#include "bfs_node.h"
#include "game_tree_node.h"

template <class Heuristic>
class AbstractSearch {
 public:
  AbstractSearch(std::shared_ptr<Heuristic> heuristic,
                 const typename Heuristic::BoardT &board)
      : heuristic(heuristic), board(board) {
    if (!heuristic)
      throw std::invalid_argument("Must pass a non-null heuristic!");
    this->heuristic->start_search();
  }

  virtual bool advance_search() = 0;

  void complete_search() {
    while (!advance_search()) {
    }
  }

  virtual ~AbstractSearch() {
    if (heuristic) {
      heuristic->complete_search();
    }
  }

  virtual std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree() = 0;

 protected:
  std::shared_ptr<Heuristic> heuristic;
  const typename Heuristic::BoardT board;
};

template <class Heuristic, class NodeT>
class Search : public AbstractSearch<Heuristic> {
 public:
  Search(std::shared_ptr<Heuristic> heuristic,
         const typename Heuristic::BoardT &board)
      : AbstractSearch<Heuristic>(heuristic, board),
        root(NodeT::create(board, heuristic->evaluate(board))) {}

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

  virtual std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree()
      override {
    return std::dynamic_pointer_cast<Node<typename Heuristic::BoardT>>(root);
  }

 protected:
  virtual std::shared_ptr<NodeT> select_next_node() {
    return std::dynamic_pointer_cast<NodeT>(root->select());
  }

  virtual bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic,
      const typename Heuristic::BoardT &board) const {
    return this->root->determined();
  }

  virtual void on_node_expansion(std::shared_ptr<NodeT> expanded_node,
                                 std::shared_ptr<Heuristic> heuristic,
                                 const typename Heuristic::BoardT &board) {}

  std::shared_ptr<NodeT> root;
};

#endif  // SEARCHES_H_INCLUDED
