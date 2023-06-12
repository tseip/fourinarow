#ifndef NINAROW_BFS_INCLUDED
#define NINAROW_BFS_INCLUDED

#include <memory>

#include "ninarow_heuristic.h"
#include "searches.h"

namespace NInARow {

template <class Heuristic>
class NInARowBestFirstSearch : public BestFirstSearch<Heuristic> {
 public:
  static std::shared_ptr<NInARowBestFirstSearch> create() {
    return std::shared_ptr<NInARowBestFirstSearch<Heuristic>>(
        new NInARowBestFirstSearch());
  }

  ~NInARowBestFirstSearch(){};

 protected:
  NInARowBestFirstSearch()
      : BestFirstSearch<Heuristic>(),
        old_best_move(),
        best_move(),
        num_repetitions(0),
        iterations(0) {}

  void clear_state() override {
    Search<Heuristic>::clear_state();
    old_best_move = typename Heuristic::BoardT::MoveT();
    best_move = typename Heuristic::BoardT::MoveT();
    num_repetitions = 0;
    iterations = 0;
  }

  bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic, Player player,
      const typename Heuristic::BoardT& board) const override {
    return iterations >= (std::size_t(1.0 / heuristic->get_gamma()) + 1) ||
           num_repetitions >= heuristic->get_stopping_thresh() ||
           BestFirstSearch<Heuristic>::stopping_conditions(heuristic, player,
                                                           board);
  }

  void on_node_expansion(std::shared_ptr<Heuristic> heuristic,
                         Player /*player*/,
                         const typename Heuristic::BoardT& /*board*/) override {
    old_best_move = best_move;
    best_move = this->root->get_best_move();
    old_best_move.board_position == best_move.board_position
        ? ++num_repetitions
        : num_repetitions = 0;
    ++iterations;
    return;
  }

 private:
  typename Heuristic::BoardT::MoveT old_best_move;
  typename Heuristic::BoardT::MoveT best_move;
  std::size_t num_repetitions;
  std::size_t iterations;
};

}  // namespace NInARow

#endif  // NINAROW_BFS_INCLUDED
