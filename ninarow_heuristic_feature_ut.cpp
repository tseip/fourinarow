#include <gtest/gtest.h>

#include "ninarow_board.h"
#include "ninarow_heuristic_feature.h"

using namespace NInARow;

/**
 * Tests HeuristicFeature implementation.
 */
TEST(NInARowHeuristicFeatureTest, TestHeuristicFeature) {
  using Board = Board<3, 3, 3>;

  /*
   * Represents the following feature:
   * x.x
   * .o.
   * o.o
   * where x is a piece, . is don't care, and o is a space.
   * If we are in this position and either o is unoccupied, we have a win.
   */
  HeuristicFeature<Board> feature{{0b000000101}, {0b101010000}, 2};

  // Player 1 starts forming the feature.
  Board board;
  board.add({0, 0, 0.0, Player::Player1});

  EXPECT_FALSE(feature.contained_in(board, Player::Player1));
  EXPECT_FALSE(feature.contained_in(board, Player::Player2));
  EXPECT_TRUE(feature.can_be_completed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player2));
  EXPECT_EQ(Board::PatternT{0b000000100},
            feature.missing_pieces(board, Player::Player1));
  EXPECT_EQ(Board::PatternT{0b000000101},
            feature.missing_pieces(board, Player::Player2));
  EXPECT_EQ(1, feature.count_pieces(board, Player::Player1));
  EXPECT_EQ(0, feature.count_pieces(board, Player::Player2));
  EXPECT_EQ(3, feature.count_spaces(board));

  // Player 2 encroaches.
  board.add({0, 2, 0.0, Player::Player2});
  EXPECT_FALSE(feature.contained_in(board, Player::Player1));
  EXPECT_FALSE(feature.contained_in(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player2));
  EXPECT_EQ(Board::PatternT{0b000000100},
            feature.missing_pieces(board, Player::Player1));
  EXPECT_EQ(Board::PatternT{0b000000001},
            feature.missing_pieces(board, Player::Player2));
  EXPECT_EQ(1, feature.count_pieces(board, Player::Player1));
  EXPECT_EQ(1, feature.count_pieces(board, Player::Player2));
  EXPECT_EQ(3, feature.count_spaces(board));

  // Remove Player 2's encroachment and replace it with Player 1.
  board.remove({0, 2, 0.0, Player::Player2});
  board.add({0, 1, 0.0, Player::Player2});
  board.add({0, 2, 0.0, Player::Player1});
  EXPECT_TRUE(feature.contained_in(board, Player::Player1));
  EXPECT_FALSE(feature.contained_in(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player2));
  EXPECT_EQ(Board::PatternT{0}, feature.missing_pieces(board, Player::Player1));
  EXPECT_EQ(Board::PatternT{0b000000101},
            feature.missing_pieces(board, Player::Player2));
  EXPECT_EQ(2, feature.count_pieces(board, Player::Player1));
  EXPECT_EQ(0, feature.count_pieces(board, Player::Player2));
  EXPECT_EQ(3, feature.count_spaces(board));

  // Cover a space in the pattern from player 2. can_be_removed should now
  // become true.
  board.add({2, 0, 0.0, Player::Player2});
  EXPECT_TRUE(feature.contained_in(board, Player::Player1));
  EXPECT_FALSE(feature.contained_in(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player2));
  EXPECT_TRUE(feature.can_be_removed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player2));
  EXPECT_EQ(Board::PatternT{0}, feature.missing_pieces(board, Player::Player1));
  EXPECT_EQ(Board::PatternT{0b000000101},
            feature.missing_pieces(board, Player::Player2));
  EXPECT_EQ(2, feature.count_pieces(board, Player::Player1));
  EXPECT_EQ(0, feature.count_pieces(board, Player::Player2));
  EXPECT_EQ(2, feature.count_spaces(board));

  // Cover another space. Now the feature should be inactive.
  board.add({1, 1, 0.0, Player::Player1});
  EXPECT_FALSE(feature.contained_in(board, Player::Player1));
  EXPECT_FALSE(feature.contained_in(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_completed(board, Player::Player2));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player1));
  EXPECT_FALSE(feature.can_be_removed(board, Player::Player2));
  EXPECT_EQ(Board::PatternT{0}, feature.missing_pieces(board, Player::Player1));
  EXPECT_EQ(Board::PatternT{0b000000101},
            feature.missing_pieces(board, Player::Player2));
  EXPECT_EQ(2, feature.count_pieces(board, Player::Player1));
  EXPECT_EQ(0, feature.count_pieces(board, Player::Player2));
  EXPECT_EQ(1, feature.count_spaces(board));
}
