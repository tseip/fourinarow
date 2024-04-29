#ifndef NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
#define NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED

#include <Eigen/Dense>
#include <unordered_map>

#include "ninarow_heuristic_feature.h"
#include "player.h"

namespace NInARow {

/**
 * Counts the number of overlapping bits between a given bitset and a vector of
 * known bitsets in an efficient, vectorized way. Uses size_t instead of actual
 * bits to keep track of total overlap counts in the final evaluation.
 *
 * @tparam N The maximum length of all of the bitsets in the known vector of
 * bitsets.
 */
template <std::size_t N>
class VectorizedBitsetCounter {
 private:
  /**
   * Represents the known vector of bitsets - an M dimensional vector of length
   * N, where M is the number of bitsets that have been registered for
   * evaluation.
   */
  Eigen::Matrix<std::size_t, Eigen::Dynamic, N> bitset_matrix;

  /**
   * Converts a bitset to a one-dimensional vector of size_ts
   *
   * @param bitset The set of bits to convert.
   *
   * @return A vector of size_t, where each set element of the bitset
   * corresponds to a 1 in the vector.
   */
  static Eigen::Vector<std::size_t, N> bitset_to_vector(
      const std::bitset<N> &bitset) {
    Eigen::Vector<std::size_t, N> vector;
    for (std::size_t i = 0; i < N; ++i) {
      vector(i) = static_cast<std::size_t>(bitset[i]);
    }
    return vector;
  }

 public:
  /**
   * Constructor.
   */
  VectorizedBitsetCounter() : bitset_matrix(0, N) {}

  /**
   * Adds a bitset into our known pool. After this function is called, each
   * query will return an additional line representing the bit overlap count
   * with this bitset.
   *
   * @param bitset The bitset to add.
   */
  void register_bitset(const std::bitset<N> &bitset) {
    bitset_matrix.conservativeResize(bitset_matrix.rows() + 1, Eigen::NoChange);
    bitset_matrix.row(bitset_matrix.rows() - 1) = bitset_to_vector(bitset);
  }

  /**
   * Queries all of the added bitsets against a new bitset. Returns a vector
   * where each element of the vector corresponds to a count of the overlapping
   * bits between each line of our registered bitsets and the given bitset.
   *
   * @param bitset The bitset to query against.
   *
   * @return A list of bit overlap counts, where each element corresponds to the
   * bit overlap count for each registered bitset against the given bitset.
   */
  std::vector<std::size_t> query(std::bitset<N> bitset) const {
    const Eigen::Vector<std::size_t, Eigen::Dynamic> count_results =
        bitset_matrix * bitset_to_vector(bitset);
    return {count_results.data(),
            count_results.data() + count_results.rows() * count_results.cols()};
  }
};

/**
 * Registers a number of features that can all be evaluated simultaneously and
 * efficiently on given boards.
 *
 * @tparam Board The board that the feature will evaluate.
 */
template <typename Board>
class VectorizedFeatureEvaluator {
 private:
  /**
   * The number of features we're tracking.
   */
  std::size_t feature_count;

  /**
   * A counter representing the set of all of the pieces corresponding to all of
   * the features we're tracking. (A feature comprises pieces and spaces.) Each
   * line of this counter represents one feature's pieces.
   */
  VectorizedBitsetCounter<Board::get_board_size()> feature_pieces_bitsets;

  /**
   * A counter representing the set of all of the spaces corresponding to all of
   * the features we're tracking. (A feature comprises pieces and spaces.) Each
   * line of this counter represents one feature's spaces.
   */
  VectorizedBitsetCounter<Board::get_board_size()> feature_spaces_bitsets;

 public:
  /**
   * Constructor.
   */
  VectorizedFeatureEvaluator()
      : feature_count(0), feature_pieces_bitsets(), feature_spaces_bitsets() {}

  /**
   * Adds a new feature to the evaluator.
   *
   * @param feature The feature to add.
   *
   * @return The total number of features this evaluator is tracking.
   */
  std::size_t register_feature(const HeuristicFeature<Board> &feature) {
    feature_pieces_bitsets.register_bitset(feature.pieces.positions);
    feature_spaces_bitsets.register_bitset(feature.spaces.positions);
    return feature_count++;
  }

  /**
   * Given a board and a player, count the number of pieces that the player
   * has on the board which overlap with each of our registered features'
   * pieces.
   *
   * @param b The board to evaluate.
   * @param player The player whose pieces we are evaluating.
   *
   * @return A list of counts representing the number of pieces that the
   * player has on the board that overlap with each feature in order.
   */
  std::vector<std::size_t> query_pieces(const Board &b, Player player) const {
    return feature_pieces_bitsets.query(b.get_pieces(player).positions);
  }

  /**
   * Given a board, count the number of spaces on the board which overlap
   * with each of our registered features' spaces.
   *
   * @param b The board to evaluate.
   *
   * @return A list of counts representing the amount of overlap between
   * between the board's spaces and each feature's spaces.
   */
  std::vector<std::size_t> query_spaces(const Board &b) const {
    return feature_spaces_bitsets.query(b.get_spaces().positions);
  }
};
}  // namespace NInARow

#endif  // NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
