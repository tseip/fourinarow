#ifndef SEARCHES_H_INCLUDED
#define SEARCHES_H_INCLUDED

#include <iostream>
#include <memory>

#include "bfs_node.h"
#include "game_tree_node.h"
#include "player.h"

template <class Heuristic>
class Search {
 public:
  typename Heuristic::BoardT::MoveT search(
      std::shared_ptr<Heuristic> heuristic, Player player,
      const typename Heuristic::BoardT& board) {
    begin_search(heuristic, player, board);
    while (!dispatch()) {
    }

    return root->get_best_move();
  }

  bool dispatch() {
    if (!heuristic) return true;
    if (stopping_conditions(heuristic, player, board)) {
      this->heuristic->complete_search();
      return true;
    } else {
      current_node = root->select();
      const std::vector<typename Heuristic::BoardT::MoveT> candidate_moves =
          heuristic->get_pruned_moves(current_node->get_board(), player);
      current_node->expand(candidate_moves);
      on_node_expansion(heuristic, player, board);
      return false;
    }
  }

  void begin_search(std::shared_ptr<Heuristic> heuristic, Player player,
                    const typename Heuristic::BoardT& board) {
    clear_state();
    this->heuristic = heuristic;
    this->player = player;
    this->board = board;
    this->heuristic->start_search();
    set_root(heuristic, player, board);
  }

  std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree() { return root; }

 protected:
  Search() : root(), current_node(), heuristic(), player(), board() {}

  virtual ~Search() {
    if (heuristic) heuristic->complete_search();
  }

  virtual void clear_state() = 0;
  virtual void set_root(std::shared_ptr<Heuristic> heuristic, Player player,
                        const typename Heuristic::BoardT& board) = 0;
  virtual bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic, Player player,
      const typename Heuristic::BoardT& board) const = 0;
  virtual void on_node_expansion(std::shared_ptr<Heuristic> heuristic,
                                 Player player,
                                 const typename Heuristic::BoardT& board) = 0;

 protected:
  std::shared_ptr<Node<typename Heuristic::BoardT>> root;
  std::shared_ptr<Node<typename Heuristic::BoardT>> current_node;

  std::shared_ptr<Heuristic> heuristic;
  Player player;
  typename Heuristic::BoardT board;
};

template <class Heuristic>
class BestFirstSearch : public Search<Heuristic> {
 public:
  static std::shared_ptr<BestFirstSearch> create() {
    return std::shared_ptr<BestFirstSearch>(new BestFirstSearch());
  }

  ~BestFirstSearch(){};

 protected:
  BestFirstSearch() : Search<Heuristic>() {}

  void clear_state() override{};

  void set_root(std::shared_ptr<Heuristic> heuristic, Player /*player*/,
                const typename Heuristic::BoardT& b) override {
    this->root =
        BFSNode<typename Heuristic::BoardT>::create(b, heuristic->evaluate(b));
  }

  bool stopping_conditions(
      std::shared_ptr<Heuristic> /*heuristic*/, Player /*player*/,
      const typename Heuristic::BoardT& /*board*/) const override {
    return std::dynamic_pointer_cast<BFSNode<typename Heuristic::BoardT>>(
               this->root)
        ->determined();
  }

  void on_node_expansion(std::shared_ptr<Heuristic> /*heuristic*/,
                         Player /*player*/,
                         const typename Heuristic::BoardT& /*board*/) override {
  }
};

#endif  // SEARCHES_H_INCLUDED
