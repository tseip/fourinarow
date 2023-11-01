#ifndef NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
#define NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED

#include <Eigen/Dense>
#include <unordered_map>

#include "ninarow_heuristic_feature.h"
#include "player.h"

namespace NInARow {

template <std::size_t N>
class VectorizedBitsetCounter {
 private:
  Eigen::Matrix<std::size_t, Eigen::Dynamic, N> bitset_matrix;

  static Eigen::Vector<std::size_t, N> bitset_to_vector(
      const std::bitset<N> &bitset) {
    Eigen::Vector<std::size_t, N> vector;
    for (std::size_t i = 0; i < N; ++i) {
      vector(i) = static_cast<std::size_t>(bitset[i]);
    }
    return vector;
  }

 public:
  VectorizedBitsetCounter() : bitset_matrix(0, N) {}

  void register_bitset(const std::bitset<N> &bitset) {
    bitset_matrix.conservativeResize(bitset_matrix.rows() + 1, Eigen::NoChange);
    bitset_matrix.row(bitset_matrix.rows() - 1) = bitset_to_vector(bitset);
  }

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
  std::size_t feature_count;
  VectorizedBitsetCounter<Board::get_board_size()> feature_pieces_bitsets;
  VectorizedBitsetCounter<Board::get_board_size()> feature_spaces_bitsets;

 public:
  VectorizedFeatureEvaluator()
      : feature_count(0), feature_pieces_bitsets(), feature_spaces_bitsets() {}

  std::size_t register_feature(const HeuristicFeature<Board> &feature) {
    feature_pieces_bitsets.register_bitset(feature.pieces.positions);
    feature_spaces_bitsets.register_bitset(feature.spaces.positions);
    return feature_count++;
  }

  std::vector<std::size_t> query_pieces(const Board &b, Player player) const {
    return feature_pieces_bitsets.query(b.get_pieces(player).positions);
  }

  std::vector<std::size_t> query_spaces(const Board &b) const {
    return feature_spaces_bitsets.query(b.get_spaces().positions);
  }
};
}  // namespace NInARow

#endif  // NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
