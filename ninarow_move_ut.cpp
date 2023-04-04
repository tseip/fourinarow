#include <gtest/gtest.h>

#include "ninarow_move.h"

using namespace NInARow;

/**
 * Tests Move implementation.
 */
TEST(NInARowMoveTest, TestMove) {
  using Move = Move<3, 3, 3>;

  Move move1 = Move();
  Move move2 = Move(1, 1000, Player::Player1);
  Move move3 = Move(8, -1000, Player::Player2);
  Move move4 = Move(2, 2, -50, Player::Player1);

  // Can't construct moves that are off the board.
  EXPECT_THROW(({ Move(9, -1000, Player::Player2); }), std::out_of_range);
  EXPECT_THROW(({ Move(3, 3, -1000, Player::Player2); }), std::out_of_range);

  EXPECT_FALSE(move1 < move1);
  EXPECT_TRUE(move1 < move2);
  EXPECT_FALSE(move2 < move1);
  EXPECT_TRUE(move3 < move2);
  EXPECT_FALSE(move2 < move3);
  EXPECT_TRUE(move3 < move1);
  EXPECT_FALSE(move1 < move3);
  EXPECT_TRUE(move3 < move4);
  EXPECT_FALSE(move4 < move3);
  EXPECT_TRUE(move4 < move2);
  EXPECT_FALSE(move2 < move4);
  EXPECT_TRUE(move4 < move1);
  EXPECT_FALSE(move1 < move4);

  EXPECT_EQ(Move(0, 2, 0, Player::Player1).board_position,
            Move(2, 0, Player::Player1).board_position);
  EXPECT_EQ(Move(1, 2, 0, Player::Player1).board_position,
            Move(5, 0, Player::Player1).board_position);
  EXPECT_EQ(Move(2, 2, 0, Player::Player1).board_position,
            Move(8, 0, Player::Player1).board_position);
  EXPECT_EQ(Move(2, 0, 0, Player::Player1).board_position,
            Move(6, 0, Player::Player1).board_position);
  EXPECT_EQ(Move(2, 1, 0, Player::Player1).board_position,
            Move(7, 0, Player::Player1).board_position);
}
