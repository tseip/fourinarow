#include <gtest/gtest.h>

#include "bfs_node.h"
#include "ninarow_board.h"

/**
 * Tests game tree creation.
 */
TEST(BFSNodeTest, TestCreate) {
  using Board = NInARow::Board<3, 3, 3>;
  auto game_tree = BFSNode<Board>::create(Board(), 0.0);
  ASSERT_EQ(game_tree->get_num_leaves(), 1);
  ASSERT_FALSE(game_tree->determined());

  // Provide an invalid move.
  std::vector<Board::MoveT> bad_moves = {
      Board::MoveT(1, 1, 0.0, Player::Player2)};
  EXPECT_THROW(game_tree->expand(bad_moves), std::logic_error);

  // Provide good moves.
  std::vector<Board::MoveT> moves;
  moves.emplace_back(0, 0, 0.0, Player::Player1);
  moves.emplace_back(0, 1, 0.0, Player::Player1);
  moves.emplace_back(0, 2, 0.0, Player::Player1);
  moves.emplace_back(1, 0, 0.0, Player::Player1);
  moves.emplace_back(1, 1, 0.0, Player::Player1);
  moves.emplace_back(1, 2, 0.0, Player::Player1);
  moves.emplace_back(2, 0, 0.0, Player::Player1);
  moves.emplace_back(2, 1, 0.0, Player::Player1);
  moves.emplace_back(2, 2, 0.0, Player::Player1);
  game_tree->expand(moves);

  // Provide more invalid moves.
  bad_moves.emplace_back(0, 0, 0.0, Player::Player1);
  EXPECT_THROW(game_tree->expand(bad_moves), std::logic_error);

  ASSERT_EQ(game_tree->get_num_leaves(), 9);
  ASSERT_FALSE(game_tree->determined());

  const auto best_move = Board::MoveT(0, 0, 0.0, Player::Player1);
  ASSERT_EQ(game_tree->get_best_move().board_position,
            best_move.board_position);
}

/**
 * Tests game tree node counting functions.
 */
TEST(BFSNodeTest, TestNodeCountingFunctions) {
  using Board = NInARow::Board<3, 3, 3>;
  auto game_tree = BFSNode<Board>::create(Board(), 0.0);
  ASSERT_EQ(game_tree->get_num_leaves(), 1);
  ASSERT_FALSE(game_tree->determined());

  std::vector<Board::MoveT> moves;
  moves.emplace_back(0, 0, 1.0, Player::Player1);
  moves.emplace_back(0, 1, 0.0, Player::Player1);
  moves.emplace_back(0, 2, -1.0, Player::Player1);
  game_tree->expand(moves);

  ASSERT_EQ(game_tree->get_num_leaves(), 3);
  ASSERT_FALSE(game_tree->determined());

  for (auto child : game_tree->get_children()) {
    moves.clear();
    moves.emplace_back(1, 0, -3.0, Player::Player2);
    moves.emplace_back(1, 1, -2.0, Player::Player2);
    moves.emplace_back(1, 2, -1.0, Player::Player2);
    child->expand(moves);

    auto grandchildren = child->get_children();
    for (size_t i = 0; i < grandchildren.size(); ++i) {
      moves.clear();
      for (size_t j = 0; j < i; ++j) {
        moves.emplace_back(2, j, 1.0, Player::Player1);
      }
      grandchildren[i]->expand(moves);
    }
  }

  // The tree now has a depth of 4, with 3 nodes at depth 2,
  // 9 nodes at depth 3, and 9 nodes at depth 4. There are 1 + 3 + 9 + 9 =
  // 22 nodes in total. Since there are 3 leaf nodes at
  // depth 3 and 9 at depth 4, the mean should be 3.75.
  ASSERT_EQ(game_tree->get_node_count(), 22U);
  ASSERT_EQ(game_tree->get_num_leaves(), 12U);
  ASSERT_EQ(game_tree->get_num_internal_nodes(), 10U);
  ASSERT_EQ(game_tree->get_mean_depth(), 3.75);

  // The average branching factor of the tree is the number of nodes in the tree
  // (not counting the root) divided by the number of internal nodes in the
  // tree, i.e., 21 / 10 = 2.1.
  ASSERT_EQ(game_tree->get_average_branching_factor(), 2.1);

  // The best path through the tree for player 1 leads us to a position at depth
  // 4, and 4 - 1 - 1 = 2;
  ASSERT_EQ(game_tree->get_depth_of_pv(), 2U);
}

/**
 * Tests getting the best move from an example simple game tree.
 */
TEST(BFSNodeTest, TestGetBestMove) {
  using Board = NInARow::Board<1, 3, 2>;
  auto game_tree = BFSNode<Board>::create(Board(), 0.0);
  ASSERT_EQ(game_tree->get_num_leaves(), 1);
  ASSERT_FALSE(game_tree->determined());

  // Construct the entire game tree for this super simple game.
  const std::vector<Board::MoveT> first_moves{
      Board::MoveT(0, 0, 0.0, Player::Player1),
      Board::MoveT(0, 1, 1.0, Player::Player1),
      Board::MoveT(0, 2, 0.0, Player::Player1)};
  game_tree->expand(first_moves);

  const auto& children = game_tree->get_children();
  for (size_t i = 0; i < children.size(); ++i) {
    std::vector<Board::MoveT> second_moves;
    for (size_t j = 0; j < first_moves.size(); ++j) {
      if (j == i) continue;
      second_moves.emplace_back(0, j, (j == 1) ? 0.0 : 1.0, Player::Player2);
    }
    children[i]->expand(second_moves);

    const auto& grandchildren = children[i]->get_children();
    for (size_t j = 0; j < grandchildren.size(); ++j) {
      const std::vector<Board::MoveT> third_move{
          Board::MoveT(0, 3 - (i + grandchildren[j]->get_move().board_position),
                       i == 1 ? 1.0 : 0.0, Player::Player1)};
      grandchildren[j]->expand(third_move);
    }
  }

  ASSERT_TRUE(game_tree->determined());

  for (auto& node : *game_tree) {
    switch (node.get_depth()) {
      case 1:
        // Player 1 should play in the middle of the board in order to win.
        ASSERT_EQ(node.get_best_move().board_position, 1U);
        break;
      case 2:
        // Player 2 should play in the middle of the board if they can,
        // otherwise they can play anywhere. Default to the first possible move.
        ASSERT_EQ(node.get_best_move().board_position,
                  node.get_move().board_position == 1U ? 0U : 1U);
        break;
      case 3:
        // Player 1 must play in the remaining spot.
        ASSERT_EQ(node.get_best_move().board_position,
                  3 - (node.get_move().board_position +
                       node.get_parent()->get_move().board_position));
        break;
      case 4:
        // There are no more moves available.
        ASSERT_THROW(node.get_best_move(), std::logic_error);
        break;
      default:
        FAIL() << "This isn't a valid depth.";
    }
  }
}
