#include <gtest/gtest.h>

#include "ninarow_board.h"

using namespace NInARow;

/**
 * Tests Board implementation.
 */
TEST(NInARowBoardTest, TestBoard) {
  using Board = Board<3, 3, 3>;

  Board board;
  EXPECT_EQ(board.active_player(), Player::Player1);
  EXPECT_EQ(board.num_pieces(), 0);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));
  EXPECT_FALSE(board.game_is_drawn());

  // Add a move, check the board state.
  Board::MoveT move1(0, 0, 0.0, Player::Player1);
  board.add(move1);
  EXPECT_EQ(board.active_player(), Player::Player2);
  EXPECT_EQ(board.num_pieces(), 1);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));
  EXPECT_FALSE(board.game_is_drawn());

  // We can't add a move where a move already exists.
  EXPECT_THROW((board.add(Board::MoveT(0, 0, 0.0, Player::Player2))),
               std::invalid_argument);

  // Add a second move.
  Board::MoveT move2(0, 2, 0.0, Player::Player2);
  board = board + move2;
  EXPECT_EQ(board.active_player(), Player::Player1);
  EXPECT_EQ(board.num_pieces(), 2);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));
  EXPECT_FALSE(board.game_is_drawn());

  // Let player 1 win.
  board.add(Board::MoveT(1, 0, 0.0, Player::Player1));
  EXPECT_EQ(board.num_pieces(), 3);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  board.add(Board::MoveT(1, 1, 0.0, Player::Player2));
  EXPECT_EQ(board.num_pieces(), 4);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  board.add(Board::MoveT(2, 0, 0.0, Player::Player1));
  EXPECT_EQ(board.num_pieces(), 5);
  EXPECT_TRUE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  // Fill up the board.
  board.add(Board::MoveT(2, 1, 0.0, Player::Player2));
  EXPECT_EQ(board.num_pieces(), 6);
  board.add(Board::MoveT(0, 1, 0.0, Player::Player1));
  EXPECT_EQ(board.num_pieces(), 7);
  board.add(Board::MoveT(2, 2, 0.0, Player::Player2));
  EXPECT_EQ(board.num_pieces(), 8);
  board.add(Board::MoveT(1, 2, 0.0, Player::Player1));
  EXPECT_EQ(board.num_pieces(), 9);
  EXPECT_TRUE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  EXPECT_EQ(board.to_string(),
            "+---+\n"
            "|oox|\n"
            "|oxo|\n"
            "|oxx|\n"
            "+---+\n");
  EXPECT_FALSE(board.game_is_drawn());
  EXPECT_EQ(board.num_pieces(), 9);

  // Remove a winning piece for player 1.
  EXPECT_THROW((board.remove(Board::MoveT(0, 0, 0.0, Player::Player2))),
               std::logic_error);
  EXPECT_THROW((board.remove(Board::MoveT(0, 2, 0.0, Player::Player1))),
               std::invalid_argument);
  board.remove(Board::MoveT(0, 0, 0.0, Player::Player1));
  EXPECT_EQ(board.num_pieces(), 8);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  // Add a winning piece for player 2. First remove one of player 2's pieces.
  board.remove(Board::MoveT(0, 2, 0.0, Player::Player2));
  EXPECT_EQ(board.num_pieces(), 7);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));

  board = board + Board::MoveT(0, 0, 0.0, Player::Player2);
  EXPECT_EQ(board.num_pieces(), 8);
  EXPECT_TRUE(board.player_has_won(Player::Player2));
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.game_is_drawn());

  // Construct a drawn board.
  board = board - Board::MoveT(2, 2, 0.0, Player::Player2);
  board = board + Board::MoveT(0, 2, 0.0, Player::Player2);
  board = board + Board::MoveT(2, 2, 0.0, Player::Player1);
  EXPECT_FALSE(board.player_has_won(Player::Player1));
  EXPECT_FALSE(board.player_has_won(Player::Player2));
  EXPECT_TRUE(board.game_is_drawn());

  Board temp = board;
  board.reset();

  EXPECT_NE(board, temp);
  EXPECT_EQ(board, Board());
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
  board.add(Board::MoveT(0, 0, 0.0, Player::Player1));
  board.add(Board::MoveT(0, 2, 0.0, Player::Player2));
  board.add(Board::MoveT(0, 1, 0.0, Player::Player1));
  board.add(Board::MoveT(1, 1, 0.0, Player::Player2));
  board.add(Board::MoveT(2, 0, 0.0, Player::Player1));
  board.add(Board::MoveT(1, 2, 0.0, Player::Player2));
  board.add(Board::MoveT(2, 1, 0.0, Player::Player1));

  EXPECT_TRUE(board.contains_move(Board::MoveT(0, 0, 0.0, Player::Player1)));
  EXPECT_FALSE(board.contains_move(Board::MoveT(0, 0, 0.0, Player::Player2)));
  EXPECT_FALSE(board.contains_move(Board::MoveT(1, 0, 0.0, Player::Player1)));
  EXPECT_FALSE(board.contains_move(Board::MoveT(1, 0, 0.0, Player::Player2)));
  EXPECT_FALSE(board.contains_spaces(0));
  EXPECT_TRUE(board.contains_spaces(3));
  EXPECT_FALSE(board.contains_spaces(6));
  EXPECT_TRUE(board.contains_spaces(8));

  Board::PatternT all("111111111");
  EXPECT_TRUE(Board().contains_spaces(all));

  EXPECT_EQ(board.count_pieces(all, Player::Player1), 4U);
  EXPECT_EQ(board.count_pieces(all, Player::Player2), 3U);
  EXPECT_EQ(board.count_spaces(all), 2U);
  EXPECT_FALSE(board.contains_spaces(all));
  EXPECT_FALSE(board.contains(all, Player::Player1));
  EXPECT_FALSE(board.contains(all, Player::Player2));

  Board::PatternT none("000000000");
  // Trivially true.
  EXPECT_TRUE(Board().contains_spaces(none));

  EXPECT_EQ(board.count_pieces(none, Player::Player1), 0U);
  EXPECT_EQ(board.count_pieces(none, Player::Player2), 0U);
  EXPECT_EQ(board.count_spaces(none), 0U);
  // Trivially true.
  EXPECT_TRUE(board.contains_spaces(none));
  EXPECT_TRUE(board.contains(none, Player::Player1));
  EXPECT_TRUE(board.contains(none, Player::Player2));

  Board::PatternT some(
      "011"
      "011"
      "011");
  EXPECT_EQ(board.count_pieces(some, Player::Player1), 4U);
  EXPECT_EQ(board.count_pieces(some, Player::Player2), 1U);
  EXPECT_EQ(board.count_spaces(some), 1U);
  EXPECT_FALSE(board.contains_spaces(some));
  EXPECT_FALSE(board.contains(some, Player::Player1));
  EXPECT_FALSE(board.contains(some, Player::Player2));

  Board::PatternT just_spaces(
      "100"
      "001"
      "000");
  EXPECT_EQ(board.count_pieces(just_spaces, Player::Player1), 0U);
  EXPECT_EQ(board.count_pieces(just_spaces, Player::Player2), 0U);
  EXPECT_EQ(board.count_spaces(just_spaces), 2U);
  EXPECT_TRUE(board.contains_spaces(just_spaces));
  EXPECT_FALSE(board.contains(just_spaces, Player::Player1));
  EXPECT_FALSE(board.contains(just_spaces, Player::Player2));

  Board::PatternT some_player_one(
      "011"
      "000"
      "000");
  EXPECT_EQ(board.count_pieces(some_player_one, Player::Player1), 2U);
  EXPECT_EQ(board.count_pieces(some_player_one, Player::Player2), 0U);
  EXPECT_EQ(board.count_spaces(some_player_one), 0U);
  EXPECT_FALSE(board.contains_spaces(some_player_one));
  EXPECT_TRUE(board.contains(some_player_one, Player::Player1));
  EXPECT_FALSE(board.contains(some_player_one, Player::Player2));
}
