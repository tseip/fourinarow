#include <gtest/gtest.h>

#include "fourbynine_features.h"
#include "ninarow_bfs.h"
#include "ninarow_board.h"
#include "ninarow_heuristic.h"

using namespace NInARow;

TEST(NInARowHeuristicTest, TestHeuristicRandomMoves) {
  using Board = Board<4, 9, 4>;

  auto heuristic = Heuristic<Board>::create();
  heuristic->seed_generator(0);

  Board b;
  EXPECT_EQ(heuristic->evaluate(b), 0);
  std::size_t moves_remaining = Board::get_max_num_moves();
  while (moves_remaining-- != 0) {
    b = b + heuristic->get_random_move(b);
    EXPECT_EQ(heuristic->get_moves(b, Player::Player1).size(), moves_remaining);
  }

  EXPECT_EQ(b.num_pieces(), Board::get_max_num_moves());
}

TEST(NInARowHeuristicTest, TestHeuristicGetBestMoveBFS) {
  using Board = Board<4, 9, 4>;

  auto heuristic = Heuristic<Board>::create();
  heuristic->seed_generator(10);
  Board b;
  Player player = Player::Player1;

  std::size_t moves_remaining = Board::get_max_num_moves();
  while (moves_remaining-- != 0) {
    auto moves = heuristic->get_moves(b, player);
    std::sort(moves.begin(), moves.end(), [](const auto& m1, const auto& m2) {
      return m1.board_position < m2.board_position;
    });

    auto bfs = std::make_shared<NInARowBestFirstSearch<Heuristic<Board>>>(
        heuristic, b);
    bfs->complete_search();
    auto move = heuristic->get_best_move(bfs->get_tree());
    b = b + move;

    player = get_other_player(player);
    if (b.player_has_won(Player::Player1) ||
        b.player_has_won(Player::Player2)) {
      break;
    }
  }
}
