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

  /**
   * The number of pieces in the pattern.
   */
  std::size_t piece_count;

 public:
  /**
   * Default constructor. Makes an empty feature.
   */
  HeuristicFeature() = default;

  /**
   * Constructor.
   *
   * @param pieces The pieces in this feature.
   * @param spaces The spaces in this feature.
   * @param min_space_occupancy The minimum number of spaces that must be
   * unoccupied for this feature to be present.
   */
  HeuristicFeature(typename Board::PatternT pieces,
                   typename Board::PatternT spaces,
                   std::size_t min_space_occupancy)
      : pieces(pieces),
        spaces(spaces),
        min_space_occupancy(min_space_occupancy),
        piece_count(pieces.positions.count()) {
    if (pieces.count_overlap(spaces) != 0) {
      throw std::logic_error(
          "The supplied piece and void patterns overlap each other!");
    }
  }

  /**
   * @param b The board to check.
   * @param player The player whose pieces should be checked.
   *
   * @return The number of pieces on the given board played by the given player
   * that overlap with the pattern's pieces.
   */
  std::size_t count_pieces(const Board& b, Player player) const {
    return pieces.count_overlap(b.get_pieces(player));
  }

  /**
   * @param b The board to check.
   *
   * @return The number of empty spaces on the given board that overlap with the
   * pattern's spaces.
   */
  std::size_t count_spaces(const Board& b) const {
    return spaces.count_overlap(b.get_spaces());
  }

  /**
   * @param b The board to check.
   * @param player The player whose pieces should be checked.
   *
   * @return True if the given player has this feature, i.e. all spaces are
   * covered by the player and the min_space_occupancy constraints are
   * satisfied.
   */
  bool contained_in(const Board& b, Player player) const {
    return contained_in(count_pieces(b, player), count_spaces(b));
  }

  /**
   * @param b The board to check.
   * @param player The player whose pieces should be checked.
   *
   * @return True if the given player can make this feature in exactly one move,
   * which also means the min_space_occupancy constraints must be satisfied.
   */
  bool can_be_completed(const Board& b, Player player) const {
    return can_be_completed(count_pieces(b, player),
                            count_pieces(b, get_other_player(player)),
                            count_spaces(b));
  }

  /**
   * @param b The board to check.
   * @param player The player whose pieces should be checked.
   *
   * @return True iff the given player has this feature and it can be removed in
   * a single move, i.e. the given player's pieces fully cover this feature's
   * pieces and min_space_occupancy is exactly satisfied.
   */
  bool can_be_removed(const Board& b, Player player) const {
    return can_be_removed(count_pieces(b, player), count_spaces(b));
  }

  /**
   * This is a helper function for contained_in which allows for external
   * overlap calculators to evaluate piece overlaps and feed them into the
   * feature object directly.
   *
   * @param player_piece_count The number of pieces that the given player has in
   * this feature on some board.
   * @param open_space_count The number of open spaces that this feature has on
   * some board.
   *
   * @return True if the given player has this feature, i.e. all spaces are
   * covered by the player and the min_space_occupancy constraints are
   * satisfied.
   */
  bool contained_in(std::size_t player_piece_count,
                    std::size_t open_space_count) const {
    return player_piece_count == piece_count &&
           open_space_count >= min_space_occupancy;
  }

  /**
   * This is a helper function for contained_in which allows for external
   * overlap calculators to evaluate piece overlaps and feed them into the
   * feature object directly.
   *
   * @param player_piece_count The number of pieces that the given player has in
   * this feature on some board.
   * @param opponent_piece_count The number of pieces that the opposing player
   * has in this feature on some board.
   * @param open_space_count The number of open spaces that this feature has on
   * some board.
   *
   * @return True if the given player can make this feature in exactly one move,
   * which also means the min_space_occupancy constraints must be satisfied.
   */
  bool can_be_completed(std::size_t player_piece_count,
                        std::size_t opponent_piece_count,
                        std::size_t open_space_count) const {
    return player_piece_count + 1 == piece_count && opponent_piece_count == 0 &&
           open_space_count >= min_space_occupancy;
  }

  /**
   * This is a helper function for contained_in which allows for external
   * overlap calculators to evaluate piece overlaps and feed them into the
   * feature object directly.
   *
   * @param player_piece_count The number of pieces that the given player has in
   * this feature on some board.
   * @param open_space_count The number of open spaces that this feature has on
   * some board.
   *
   * @return True iff the given player has this feature and it can be removed in
   * a single move, i.e. the given player's pieces fully cover this feature's
   * pieces and min_space_occupancy is exactly satisfied.
   */
  bool can_be_removed(std::size_t player_piece_count,
                      std::size_t open_space_count) const {
    return player_piece_count == piece_count &&
           open_space_count == min_space_occupancy;
  }

  /**
   * Returns a pattern containing all of the pieces the given player would need
   * to play to complete this feature. Ignores the existence of opposing
   * player's pieces.
   *
   * @param b The board to check.
   * @param player The player to check.
   *
   * @return A pattern containing all of the pieces the player would have to
   * play to complete this feature.
   */
  typename Board::PatternT missing_pieces(const Board& b, Player player) const {
    return b.missing_pieces(pieces, player);
  }

  /**
   * @return A string representing this feature.
   */
  std::string to_string() const {
    std::stringstream s;
    s << pieces.to_string() << " " << spaces.to_string() << " "
      << min_space_occupancy;
    return s.str();
  }
};
}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_FEATURE_H_INCLUDED
