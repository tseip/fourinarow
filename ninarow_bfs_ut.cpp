#include <gtest/gtest.h>

#include "fourbynine_features.h"
#include "ninarow_bfs.h"
#include "ninarow_heuristic.h"

/**
 * Tests game tree creation.
 */
TEST(SearchesTest, TestCreate) {
  using Board = NInARow::Board<4, 9, 4>;
  auto bfs = BestFirstSearch<Heuristic<Board>>::create();
}
