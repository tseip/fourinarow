#include <gtest/gtest.h>

#include "ninarow_board.h"

using namespace NInARow;

/**
 * Tests Board implementation.
 */
TEST(NInARowBoardTest, TestBoard) {
  using Board = Board<3, 3, 3>;

  Board board;
  ASSERT_EQ(board.active_player(), Player::Player1);
  ASSERT_EQ(board.num_pieces(), 0);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));
  ASSERT_FALSE(board.game_is_drawn());

  // Add a move, check the board state.
  Board::Move move1(0, 0, 0.0, Player::Player1);
  board.add(move1);
  ASSERT_EQ(board.active_player(), Player::Player2);
  ASSERT_EQ(board.num_pieces(), 1);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));
  ASSERT_FALSE(board.game_is_drawn());

  // We can't add a move where a move already exists.
  EXPECT_THROW(({ board.add(Board::Move(0, 0, 0.0, Player::Player2)); }),
               std::invalid_argument);

  // Add a second move.
  Board::Move move2(0, 2, 0.0, Player::Player2);
  board = board + move2;
  ASSERT_EQ(board.active_player(), Player::Player1);
  ASSERT_EQ(board.num_pieces(), 2);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));
  ASSERT_FALSE(board.game_is_drawn());

  // Let player 1 win.
  board.add(Board::Move(1, 0, 0.0, Player::Player1));
  ASSERT_EQ(board.num_pieces(), 3);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  board.add(Board::Move(1, 1, 0.0, Player::Player2));
  ASSERT_EQ(board.num_pieces(), 4);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  board.add(Board::Move(2, 0, 0.0, Player::Player1));
  ASSERT_EQ(board.num_pieces(), 5);
  ASSERT_TRUE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  // Fill up the board.
  board.add(Board::Move(2, 1, 0.0, Player::Player2));
  ASSERT_EQ(board.num_pieces(), 6);
  board.add(Board::Move(0, 1, 0.0, Player::Player1));
  ASSERT_EQ(board.num_pieces(), 7);
  board.add(Board::Move(2, 2, 0.0, Player::Player2));
  ASSERT_EQ(board.num_pieces(), 8);
  board.add(Board::Move(1, 2, 0.0, Player::Player1));
  ASSERT_EQ(board.num_pieces(), 9);
  ASSERT_TRUE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  ASSERT_EQ(board.to_string(),
            "+---+\n"
            "|oox|\n"
            "|oxo|\n"
            "|oxx|\n"
            "+---+\n");
  ASSERT_FALSE(board.game_is_drawn());
  ASSERT_EQ(board.num_pieces(), 9);

  // Remove a winning piece for player 1.
  EXPECT_THROW(({ board.remove(Board::Move(0, 0, 0.0, Player::Player2)); }),
               std::logic_error);
  EXPECT_THROW(({ board.remove(Board::Move(0, 2, 0.0, Player::Player1)); }),
               std::invalid_argument);
  board.remove(Board::Move(0, 0, 0.0, Player::Player1));
  ASSERT_EQ(board.num_pieces(), 8);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  // Add a winning piece for player 2. First remove one of player 2's pieces.
  board.remove(Board::Move(0, 2, 0.0, Player::Player2));
  ASSERT_EQ(board.num_pieces(), 7);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));

  board = board + Board::Move(0, 0, 0.0, Player::Player2);
  ASSERT_EQ(board.num_pieces(), 8);
  ASSERT_TRUE(board.player_has_won(Player::Player2));
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.game_is_drawn());

  // Construct a drawn board.
  board = board - Board::Move(2, 2, 0.0, Player::Player2);
  board = board + Board::Move(0, 2, 0.0, Player::Player2);
  board = board + Board::Move(2, 2, 0.0, Player::Player1);
  ASSERT_FALSE(board.player_has_won(Player::Player1));
  ASSERT_FALSE(board.player_has_won(Player::Player2));
  ASSERT_TRUE(board.game_is_drawn());

  Board temp = board;
  board.reset();

  ASSERT_NE(board, temp);
  ASSERT_EQ(board, Board());
}

/**
 * Tests some board utility functions.
 */
TEST(NInARowBoardTest, TestBoardUtility) {
  using Board = Board<3, 3, 3>;
  Board board;

  // Set up the board.
  // o o x
  // . x x
  // o o .
  board.add(Board::Move(0, 0, 0.0, Player::Player1));
  board.add(Board::Move(0, 2, 0.0, Player::Player2));
  board.add(Board::Move(0, 1, 0.0, Player::Player1));
  board.add(Board::Move(1, 1, 0.0, Player::Player2));
  board.add(Board::Move(2, 0, 0.0, Player::Player1));
  board.add(Board::Move(1, 2, 0.0, Player::Player2));
  board.add(Board::Move(2, 1, 0.0, Player::Player1));

  ASSERT_TRUE(board.contains_move(Board::Move(0, 0, 0.0, Player::Player1)));
  ASSERT_FALSE(board.contains_move(Board::Move(0, 0, 0.0, Player::Player2)));
  ASSERT_FALSE(board.contains_move(Board::Move(1, 0, 0.0, Player::Player1)));
  ASSERT_FALSE(board.contains_move(Board::Move(1, 0, 0.0, Player::Player2)));
  ASSERT_FALSE(board.contains_spaces(0));
  ASSERT_TRUE(board.contains_spaces(3));
  ASSERT_FALSE(board.contains_spaces(6));
  ASSERT_TRUE(board.contains_spaces(8));

  Board::Pattern all("111111111");
  ASSERT_TRUE(Board().contains_spaces(all));

  ASSERT_EQ(board.count_pieces(all, Player::Player1), 4U);
  ASSERT_EQ(board.count_pieces(all, Player::Player2), 3U);
  ASSERT_EQ(board.count_spaces(all), 2U);
  ASSERT_FALSE(board.contains_spaces(all));
  ASSERT_FALSE(board.contains(all, Player::Player1));
  ASSERT_FALSE(board.contains(all, Player::Player2));

  Board::Pattern none("000000000");
  // Trivially true.
  ASSERT_TRUE(Board().contains_spaces(none));

  ASSERT_EQ(board.count_pieces(none, Player::Player1), 0U);
  ASSERT_EQ(board.count_pieces(none, Player::Player2), 0U);
  ASSERT_EQ(board.count_spaces(none), 0U);
  // Trivially true.
  ASSERT_TRUE(board.contains_spaces(none));
  ASSERT_TRUE(board.contains(none, Player::Player1));
  ASSERT_TRUE(board.contains(none, Player::Player2));

  Board::Pattern some(
      "011"
      "011"
      "011");
  ASSERT_EQ(board.count_pieces(some, Player::Player1), 4U);
  ASSERT_EQ(board.count_pieces(some, Player::Player2), 1U);
  ASSERT_EQ(board.count_spaces(some), 1U);
  ASSERT_FALSE(board.contains_spaces(some));
  ASSERT_FALSE(board.contains(some, Player::Player1));
  ASSERT_FALSE(board.contains(some, Player::Player2));

  Board::Pattern just_spaces(
      "100"
      "001"
      "000");
  ASSERT_EQ(board.count_pieces(just_spaces, Player::Player1), 0U);
  ASSERT_EQ(board.count_pieces(just_spaces, Player::Player2), 0U);
  ASSERT_EQ(board.count_spaces(just_spaces), 2U);
  ASSERT_TRUE(board.contains_spaces(just_spaces));
  ASSERT_FALSE(board.contains(just_spaces, Player::Player1));
  ASSERT_FALSE(board.contains(just_spaces, Player::Player2));

  Board::Pattern some_player_one(
      "011"
      "000"
      "000");
  ASSERT_EQ(board.count_pieces(some_player_one, Player::Player1), 2U);
  ASSERT_EQ(board.count_pieces(some_player_one, Player::Player2), 0U);
  ASSERT_EQ(board.count_spaces(some_player_one), 0U);
  ASSERT_FALSE(board.contains_spaces(some_player_one));
  ASSERT_TRUE(board.contains(some_player_one, Player::Player1));
  ASSERT_FALSE(board.contains(some_player_one, Player::Player2));
}
