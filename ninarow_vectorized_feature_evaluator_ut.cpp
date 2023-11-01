#include <gtest/gtest.h>

#include "ninarow_board.h"
#include "ninarow_vectorized_feature_evaluator.h"

using namespace NInARow;

/**
 * Tests HeuristicFeature implementation.
 */
TEST(NInARowHeuristicFeatureEvaluatorTest, TestHeuristicFeatureEvaluator) {
  using Board = Board<3, 3, 3>;

  VectorizedFeatureEvaluator<Board> feature_evaluator;

  std::vector<HeuristicFeature<Board>> features;

  /*
   * Represents the following feature:
   * x.x
   * .o.
   * o.o
   * where x is a piece, . is don't care, and o is a space.
   * If we are in this position and either o is unoccupied, we have a win.
   */
  features.push_back({{0b000000101}, {0b101010000}, 2});

  /*
   * Represents the following feature:
   * xox
   * ...
   * ...
   * where x is a piece, . is don't care, and o is a space.
   * If we are in this position and either o is unoccupied, we have a win.
   */
  features.push_back({{0b000000101}, {0b000000010}, 1});

  for (const auto &feature : features) {
    feature_evaluator.register_feature(feature);
  }

  auto test_feature = [&features, &feature_evaluator](
                          std::size_t i, Board &board, Player player,
                          bool has_feature, bool can_complete, bool can_remove,
                          bool can_remove_opponent) {
    const auto player_pieces = feature_evaluator.query_pieces(board, player);
    const auto opponent_pieces =
        feature_evaluator.query_pieces(board, get_other_player(player));
    const auto spaces = feature_evaluator.query_spaces(board);
    EXPECT_EQ(features[i].contained_in(player_pieces[i], spaces[i]),
              has_feature);
    EXPECT_EQ(features[i].can_be_completed(player_pieces[i], opponent_pieces[i],
                                           spaces[i]),
              can_complete);
    EXPECT_EQ(features[i].can_be_removed(player_pieces[i], spaces[i]),
              can_remove);
    EXPECT_EQ(features[i].can_be_removed(opponent_pieces[i], spaces[i]),
              can_remove_opponent);
  };

  // Initial board has nothing interesting happening.
  Board board;
  test_feature(0, board, Player::Player1, false, false, false, false);
  test_feature(1, board, Player::Player1, false, false, false, false);
  test_feature(0, board, Player::Player2, false, false, false, false);
  test_feature(1, board, Player::Player2, false, false, false, false);

  // Player 1 starts forming the feature. They can now complete it in one move.
  board.add({0, 0, 0.0, Player::Player1});
  test_feature(0, board, Player::Player1, false, true, false, false);
  test_feature(1, board, Player::Player1, false, true, false, false);
  test_feature(0, board, Player::Player2, false, false, false, false);
  test_feature(1, board, Player::Player2, false, false, false, false);

  // Player 2 encroaches.
  board.add({0, 2, 0.0, Player::Player2});
  test_feature(0, board, Player::Player1, false, false, false, false);
  test_feature(1, board, Player::Player1, false, false, false, false);
  test_feature(0, board, Player::Player2, false, false, false, false);
  test_feature(1, board, Player::Player2, false, false, false, false);

  board.remove({0, 2, 0.0, Player::Player2});
  board.add({0, 1, 0.0, Player::Player2});
  board.add({0, 2, 0.0, Player::Player1});
  test_feature(0, board, Player::Player1, true, false, false, false);
  test_feature(1, board, Player::Player1, false, false, false, false);
  test_feature(0, board, Player::Player2, false, false, false, false);
  test_feature(1, board, Player::Player2, false, false, false, false);

  // Cover a space in the pattern from player 2. We can now remove our own
  // feature with a single piece.
  board.add({2, 0, 0.0, Player::Player2});
  test_feature(0, board, Player::Player1, true, false, true, false);
  test_feature(1, board, Player::Player1, false, false, false, false);
  // Our opponent also can remove our feature with a single piece.
  test_feature(0, board, Player::Player2, false, false, false, true);
  test_feature(1, board, Player::Player2, false, false, false, false);

  // Cover another space. Now the feature should be inactive.
  board.add({1, 1, 0.0, Player::Player1});
  test_feature(0, board, Player::Player1, false, false, false, false);
  test_feature(1, board, Player::Player1, false, false, false, false);
  test_feature(0, board, Player::Player2, false, false, false, false);
  test_feature(1, board, Player::Player2, false, false, false, false);
}
