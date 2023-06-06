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
  /**
   * Pieces relevant to the feature.
   */
  typename Board::PatternT pieces;

  /**
   * Spaces (spots where no piece for either player exists) relevant to the
   * feature.
   */
  typename Board::PatternT spaces;

  /**
   * The minimum number of spaces that must be empty in order for this feature
   * to be considered active.
   */
  std::size_t min_space_occupancy;

 public:
  HeuristicFeature() = default;

  HeuristicFeature(typename Board::PatternT pieces,
                   typename Board::PatternT spaces,
                   std::size_t min_space_occupancy)
      : pieces(pieces),
        spaces(spaces),
        min_space_occupancy(min_space_occupancy) {
    if (pieces.count_overlap(spaces) != 0) {
      throw std::logic_error(
          "The supplied piece and void patterns overlap each other!");
    }
  }

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

  typename Board::PatternT missing_pieces(const Board& b, Player player) const {
    return b.missing_pieces(pieces, player);
  }

  bool contains_spaces(const typename Board::PatternT p) const {
    return spaces.contains(p.positions);
  }

  std::string to_string() {
    std::stringstream s;
    s << pieces.to_string() << " " << spaces.to_string() << " "
      << min_space_occupancy;
    return s.str();
  }
};
}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_FEATURE_H_INCLUDED
