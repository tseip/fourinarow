#ifndef NINAROW_BFS_INCLUDED
#define NINAROW_BFS_INCLUDED

#include <memory>

#include "ninarow_heuristic.h"
#include "searches.h"

namespace NInARow {

/**
 * Represents a best-first-search algorithm with customized stopping conditions
 * (with precisely the same behavior as Bas's original implementation).
 */
template <class Heuristic>
class NInARowBestFirstSearch
    : public Search<Heuristic, BFSNode<typename Heuristic::BoardT>> {
 public:
  /**
   * Constructor.
   *
   * @param heuristic The heuristic used to evaluated the board at each
   * position.
   * @param board The board position to search from.
   */
  NInARowBestFirstSearch(std::shared_ptr<Heuristic> heuristic,
                         const typename Heuristic::BoardT& board)
      : Search<Heuristic, BFSNode<typename Heuristic::BoardT>>(heuristic,
                                                               board),
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

  /**
   * @return The number of iterations the current search has performed.
   */
  std::size_t get_iterations() const { return iterations; }

  /**
   * @return The number of times the current search has returned the same
   * move as the best consecutively.
   */
  std::size_t get_num_repetitions() const { return num_repetitions; }

 protected:
  bool stopping_conditions(
      std::shared_ptr<Heuristic> heuristic,
      const typename Heuristic::BoardT& board) const override {
    return iterations >= (std::size_t(1.0 / heuristic->get_gamma()) + 1) ||
           num_repetitions >= heuristic->get_stopping_thresh() ||
           Search<Heuristic, BFSNode<typename Heuristic::BoardT>>::
               stopping_conditions(heuristic, board);
  }

  void on_node_expansion(
      std::shared_ptr<BFSNode<typename Heuristic::BoardT>> /*expanded_node*/,
      std::shared_ptr<Heuristic> heuristic,
      const typename Heuristic::BoardT& /*board*/) override {
    typename Heuristic::BoardT::MoveT old_best_move = best_move;
    best_move = this->root->get_best_move();
    old_best_move.board_position == best_move.board_position
        ? ++num_repetitions
        : num_repetitions = 0;
    ++iterations;
    return;
  }

 protected:
  /**
   * The current best known move.
   */
  typename Heuristic::BoardT::MoveT best_move;

  /**
   * The number of times the current search has returned the same
   * move as the best consecutively.
   */
  std::size_t num_repetitions;

  /**
   * The number of iterations the current search has performed.
   */
  std::size_t iterations;
};

}  // namespace NInARow

#endif  // NINAROW_BFS_INCLUDED
