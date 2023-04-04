#ifndef NINAROW_HEURISTIC_FEATURE_H_INCLUDED
#define NINAROW_HEURISTIC_FEATURE_H_INCLUDED

#include "player.h"

namespace NInARow {

/**
 * Describes a feature on an n-in-a-row board. A feature is a pattern on the
 * board which has some heuristic gameplay value.
 *
 * @tparam Board The board that the feature will evaluate
 */
template <typename Board>
class HeuristicFeature {
 public:
  static const std::size_t Nweights = 17;

  /**
   * Pieces relevant to the feature.
   */
  typename Board::Pattern pieces;

  /**
   * Spaces (spots where no piece for either player exists) relevant to the
   * feature.
   */
  typename Board::Pattern spaces;

  /**
   * Represents how important it is to play into this feature given the
   * choice, i.e., the weight of performing the action of completing the
   * feature.
   */
  double weight_act;

  /**
   * Represents how important it is for the other player to play into this
   * feature given the choice, i.e., the weight of blocking this feature from
   * being completed by the other player.
   */
  double weight_pass;

  /**
   * How likely this feature is to be dropped by the heuristic evaluator when
   * randomly pruning features. Ranges between 0 (never drop) and 1 (always
   * drop).
   */
  double drop_rate;

  /**
   * An optional weight index used by the governing heuristic to know how to
   * weight this feature.
   */
  std::size_t weight_index;

  /**
   * The minimum number of spaces that must be empty in order for this feature
   * to be considered active.
   */
  std::size_t min_space_occupancy;

  HeuristicFeature(typename Board::Pattern pieces,
                   typename Board::Pattern spaces,
                   std::size_t min_space_occupancy,
                   std::size_t weight_index = 0)
      : pieces(pieces),
        spaces(spaces),
        weight_act(0.0),
        weight_pass(0.0),
        drop_rate(0.0),
        weight_index(weight_index),
        min_space_occupancy(min_space_occupancy) {
    if (weight_index >= Nweights) {
      throw std::out_of_range(
          "Supplied index is out of range of the weight table!");
    }

    if (pieces.count_overlap(spaces) != 0) {
      throw std::logic_error(
          "The supplied piece and void patterns overlap each other!");
    }
  }

  void update_weights(double w_act, double w_pass, double delta) {
    weight_act = w_act;
    weight_pass = w_pass;
    if (drop_rate < 0.0 || drop_rate > 1.0) {
      throw std::out_of_range(
          "Drop rate is out of range, must be between 0 and 1!");
    }
    drop_rate = delta;
  }

  double diff_act_pass() const { return weight_act - weight_pass; }

  bool contained(const Board& b, Player player) const {
    return b.contains(pieces, player);
  }

  bool is_active(const Board& b, Player player) const {
    return b.count_spaces(spaces) >= min_space_occupancy &&
           b.count_pieces(pieces, get_other_player(player)) == 0;
  }

  bool just_active(const Board& b, Player player) const {
    return b.count_spaces(spaces) == min_space_occupancy &&
           b.count_pieces(pieces, get_other_player(player)) == 0;
  }

  typename Board::Pattern missing_pieces(const Board& b, Player player) const {
    return b.missing_pieces(pieces, player);
  }

  bool contains_spaces(const typename Board::Pattern p) const {
    return spaces.contains(p.positions);
  }
};
}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_FEATURE_H_INCLUDED
