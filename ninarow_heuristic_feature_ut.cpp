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
   * IF we are in this position and either o is unoccupied, we have a win.
   */
  HeuristicFeature<Board> feature{{0b000000101}, {0b101010000}, 2, 0};

  feature.update_weights(0.8, 0.2, 0.2);

  // Player 1 starts forming the feature.
  Board board;
  board.add({0, 0, 0.0, Player::Player1});

  ASSERT_FALSE(feature.contained(board, Player::Player1));
  ASSERT_FALSE(feature.contained(board, Player::Player2));
  ASSERT_TRUE(feature.is_active(board, Player::Player1));
  ASSERT_FALSE(feature.is_active(board, Player::Player2));
  ASSERT_FALSE(feature.just_active(board, Player::Player1));
  ASSERT_FALSE(feature.just_active(board, Player::Player2));
  ASSERT_EQ(Board::Pattern{0b000000100},
            feature.missing_pieces(board, Player::Player1));
  ASSERT_EQ(Board::Pattern{0b000000101},
            feature.missing_pieces(board, Player::Player2));

  // Player 2 encroaches.
  board.add({0, 2, 0.0, Player::Player2});
  ASSERT_FALSE(feature.contained(board, Player::Player1));
  ASSERT_FALSE(feature.contained(board, Player::Player2));
  ASSERT_FALSE(feature.is_active(board, Player::Player1));
  ASSERT_FALSE(feature.is_active(board, Player::Player2));
  ASSERT_FALSE(feature.just_active(board, Player::Player1));
  ASSERT_FALSE(feature.just_active(board, Player::Player2));
  ASSERT_EQ(Board::Pattern{0b000000100},
            feature.missing_pieces(board, Player::Player1));
  ASSERT_EQ(Board::Pattern{0b000000001},
            feature.missing_pieces(board, Player::Player2));

  board.remove({0, 2, 0.0, Player::Player2});
  board.add({0, 1, 0.0, Player::Player2});
  board.add({0, 2, 0.0, Player::Player1});
  ASSERT_TRUE(feature.contained(board, Player::Player1));
  ASSERT_FALSE(feature.contained(board, Player::Player2));
  ASSERT_TRUE(feature.is_active(board, Player::Player1));
  ASSERT_FALSE(feature.is_active(board, Player::Player2));
  ASSERT_FALSE(feature.just_active(board, Player::Player1));
  ASSERT_FALSE(feature.just_active(board, Player::Player2));
  ASSERT_EQ(Board::Pattern{0}, feature.missing_pieces(board, Player::Player1));
  ASSERT_EQ(Board::Pattern{0b000000101},
            feature.missing_pieces(board, Player::Player2));

  // Cover a space in the pattern from player 2. just_active should now become
  // true.
  board.add({2, 0, 0.0, Player::Player2});
  ASSERT_TRUE(feature.contained(board, Player::Player1));
  ASSERT_FALSE(feature.contained(board, Player::Player2));
  ASSERT_TRUE(feature.is_active(board, Player::Player1));
  ASSERT_FALSE(feature.is_active(board, Player::Player2));
  ASSERT_TRUE(feature.just_active(board, Player::Player1));
  ASSERT_FALSE(feature.just_active(board, Player::Player2));
  ASSERT_EQ(Board::Pattern{0}, feature.missing_pieces(board, Player::Player1));
  ASSERT_EQ(Board::Pattern{0b000000101},
            feature.missing_pieces(board, Player::Player2));

  // Cover another space. Now the feature should be inactive.
  board.add({1, 1, 0.0, Player::Player1});
  ASSERT_TRUE(feature.contained(board, Player::Player1));
  ASSERT_FALSE(feature.contained(board, Player::Player2));
  ASSERT_FALSE(feature.is_active(board, Player::Player1));
  ASSERT_FALSE(feature.is_active(board, Player::Player2));
  ASSERT_FALSE(feature.just_active(board, Player::Player1));
  ASSERT_FALSE(feature.just_active(board, Player::Player2));
  ASSERT_EQ(Board::Pattern{0}, feature.missing_pieces(board, Player::Player1));
  ASSERT_EQ(Board::Pattern{0b000000101},
            feature.missing_pieces(board, Player::Player2));

  // Test contains_spaces.
  ASSERT_TRUE(feature.contains_spaces(Board::Pattern{0}));
  ASSERT_TRUE(feature.contains_spaces(Board::Pattern{0b000010000}));
  ASSERT_TRUE(feature.contains_spaces(Board::Pattern{0b001010000}));
  ASSERT_TRUE(feature.contains_spaces(Board::Pattern{0b101010000}));

  ASSERT_FALSE(feature.contains_spaces(Board::Pattern{0b111010000}));
  ASSERT_FALSE(feature.contains_spaces(Board::Pattern{0b010000000}));
}
