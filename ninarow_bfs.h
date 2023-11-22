#ifndef NINAROW_BFS_INCLUDED
#define NINAROW_BFS_INCLUDED

#include <memory>

#include "ninarow_heuristic.h"
#include "searches.h"

namespace NInARow {

template <class Heuristic>
class NInARowBestFirstSearch
    : public Search<Heuristic, BFSNode<typename Heuristic::BoardT>> {
 public:
  NInARowBestFirstSearch(std::shared_ptr<Heuristic> heuristic, Player player,
                         const typename Heuristic::BoardT& board)
      : Search<Heuristic, BFSNode<typename Heuristic::BoardT>>(heuristic,
                                                               player, board),
        old_best_move(),
        best_move(),
        num_repetitions(0),
        iterations(0) {}
  ~NInARowBestFirstSearch() = default;

  /**
   * @note Due to limitations of SWIG's inheritance system, we have to redefine
   * the public interface of any classes we expect to directly subclass in
   * Python. They can simply be defined as pure pass-throughs.
   *
   * @{
   */
  virtual bool advance_search() override {
    return Search<Heuristic,
                  BFSNode<typename Heuristic::BoardT>>::advance_search();
  }

  virtual std::shared_ptr<Node<typename Heuristic::BoardT>> get_tree()
      override {
    return Search<Heuristic, BFSNode<typename Heuristic::BoardT>>::get_tree();
  }
  /**
   * @}
   */

  std::size_t get_iterations() const { return iterations; }

  std::size_t get_num_repetitions() const { return num_repetitions; }

 protected:
  bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic, Player player,
      const typename Heuristic::BoardT& board) const override {
    return iterations >= (std::size_t(1.0 / heuristic->get_gamma()) + 1) ||
           num_repetitions >= heuristic->get_stopping_thresh() ||
           Search<Heuristic, BFSNode<typename Heuristic::BoardT>>::
               stopping_conditions(heuristic, player, board);
  }

  void on_node_expansion(
      std::shared_ptr<BFSNode<typename Heuristic::BoardT>> /*expanded_node*/,
      std::shared_ptr<Heuristic> heuristic, Player /*player*/,
      const typename Heuristic::BoardT& /*board*/) override {
    old_best_move = best_move;
    best_move = this->root->get_best_move();
    old_best_move.board_position == best_move.board_position
        ? ++num_repetitions
        : num_repetitions = 0;
    ++iterations;
    return;
  }

 protected:
  typename Heuristic::BoardT::MoveT old_best_move;
  typename Heuristic::BoardT::MoveT best_move;
  std::size_t num_repetitions;
  std::size_t iterations;
};

}  // namespace NInARow

#endif  // NINAROW_BFS_INCLUDED
