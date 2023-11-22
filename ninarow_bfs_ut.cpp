#include <gtest/gtest.h>

#include "fourbynine_features.h"
#include "ninarow_bfs.h"
#include "ninarow_heuristic.h"

/**
 * Tests game tree creation.
 */
TEST(SearchesTest, TestCreate) {
  using namespace NInARow;
  using Board = Board<4, 9, 4>;
  auto heuristic = Heuristic<Board>::create();
  Board board;
  auto bfs = NInARowBestFirstSearch<Heuristic<Board>>(
      heuristic, board.active_player(), board);
}
