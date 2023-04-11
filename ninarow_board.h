#ifndef NINAROW_BOARD_H_INCLUDED
#define NINAROW_BOARD_H_INCLUDED

#include <bitset>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ninarow_move.h"
#include "ninarow_pattern.h"
#include "player.h"

namespace NInARow {

/**
 * Represents a board of N-in-a-row. In a game of N in a row, players alternate
 * placing stones on a rectangular board. When a player places N stones in a
 * row, either orthogonally or diagonally, that player wins.
 *
 * @tparam HEIGHT The height of the board.
 * @tparam WIDTH The width of the board.
 * @tparam N The number of stones in a row required to win.
 */
template <std::size_t HEIGHT, std::size_t WIDTH, std::size_t N>
class Board {
 public:
  using Pattern = Pattern<HEIGHT, WIDTH, N>;
  using PatternHasher = PatternHasher<Pattern>;
  using Move = Move<HEIGHT, WIDTH, N>;

  /**
   * Getters for board dimensions.
   *
   * @{
   */
  static constexpr std::size_t get_board_height() { return HEIGHT; }
  static constexpr std::size_t get_board_width() { return WIDTH; }
  static constexpr std::size_t get_board_size() { return HEIGHT * WIDTH; }
  /**
   * @}
   */

  /**
   * @return The maximum number of moves of any possible game played on this
   * board.
   */
  static constexpr std::size_t get_max_num_moves() { return get_board_size(); }

  static_assert(get_board_size() >= 2 * N - 1,
                "N in a row boards must be large enough to accommodate a "
                "possible win, at least for the first player.");

 private:
  /**
   * Represents the pieces currently placed on the game board for both players.
   * Each player has their own pattern to represent their pieces over the game
   * board.
   *
   * Note that this representation allows for invalid game states to be
   * represented, if, e.g., both players have a piece in the same location.
   * We detect and throw on invalid game states as the board is being played
   * out.
   */
  Pattern pieces[2];

 public:
  /**
   * Default constructor.
   */
  Board() = default;

  /**
   * Constructs a board with the given piece pattern.
   *
   * @param black_pieces The black pieces on the board.
   * @param white_pieces The white pieces on the board.
   */
  Board(const Pattern &black_pieces, const Pattern &white_pieces) {
    const auto piece_diff =
        black_pieces.positions.count() - white_pieces.positions.count();
    if (black_pieces.count_overlap(white_pieces) != 0 || piece_diff > 1U ||
        piece_diff < 0U) {
      throw std::logic_error("Given board state is illegal!");
    }
    pieces[0] = black_pieces;
    pieces[1] = white_pieces;
  }

  /**
   * Clears the board.
   */
  void reset() {
    pieces[static_cast<size_t>(Player::Player1)].positions.reset();
    pieces[static_cast<size_t>(Player::Player2)].positions.reset();
  }

  /**
   * @param player The player to check.
   *
   * @return True if the given player has won, false otherwise.
   */
  bool player_has_won(const Player player) const {
    return pieces[static_cast<size_t>(player)].contains_win();
  }

  /**
   * @return True if the game is drawn, false otherwise.
   */
  bool game_is_drawn() const {
    // The game is drawn if both players' pieces cover the game board.
    return (pieces[static_cast<size_t>(Player::Player1)].positions |
            pieces[static_cast<size_t>(Player::Player2)].positions)
               .all() &&
           !player_has_won(Player::Player1) && !player_has_won(Player::Player2);
  }

  /**
   * @return True if the game is over - either one player has one, or the game
   * is drawn.
   */
  bool game_has_ended() const {
    return player_has_won(Player::Player1) || player_has_won(Player::Player2) ||
           game_is_drawn();
  }

  /**
   * @return The number of pieces on the board from both players.
   */
  std::size_t num_pieces() const {
    return (pieces[static_cast<size_t>(Player::Player1)].positions |
            pieces[static_cast<size_t>(Player::Player2)].positions)
        .count();
  }

  Player active_player() const {
    // Player 1 plays first, and play proceeds in alternating turns.
    return (num_pieces() % 2) == 0 ? Player::Player1 : Player::Player2;
  }

  std::string to_string() const { return to_string(Pattern()); }

  /**
   * @param p A pattern to highlight on the board.
   *
   * @return A string representing the board with specific board positions
   * highlighted.
   */
  std::string to_string(const Pattern p) const {
    std::stringstream stream;
    stream << "+";
    for (std::size_t col = 0; col < WIDTH; ++col) stream << "-";
    stream << "+" << std::endl;
    for (std::size_t row = 0; row < HEIGHT; ++row) {
      stream << "|";
      for (std::size_t col = 0; col < WIDTH; ++col) {
        const size_t position = row * WIDTH + col;
        if (p.positions.test(position))
          stream << "#";
        else if (pieces[static_cast<size_t>(Player::Player1)].positions.test(
                     position))
          stream << "o";
        else if (pieces[static_cast<size_t>(Player::Player2)].positions.test(
                     position))
          stream << "x";
        else
          stream << " ";
      }
      stream << "|" << std::endl;
    }
    stream << "+";
    for (unsigned int col = 0; col < WIDTH; col++) stream << "-";
    stream << "+" << std::endl;
    return stream.str();
  }

  /**
   * @param p The pattern to check.
   *
   * @return The number of positions that the given player has covered in the
   * given pattern.
   */
  std::size_t count_pieces(const Pattern p, const Player player) const {
    return pieces[static_cast<size_t>(player)].count_overlap(p);
  }

  /**
   * @param p The pattern to check.
   *
   * @return The number of positions that neither player has covered in the
   * given pattern.
   */
  std::size_t count_spaces(const Pattern p) const {
    return Pattern(pieces[static_cast<size_t>(Player::Player1)].positions |
                   pieces[static_cast<size_t>(Player::Player2)].positions)
        .count_spaces(p);
  }

  /**
   * @param pattern A pattern of pieces to cover.
   * @param player The player who desires to cover the given pattern.
   *
   * @return A pattern representing the pieces needed to be played by the
   * given player in order to cover the given pattern.
   */
  Board::Pattern missing_pieces(const Board::Pattern &pattern,
                                Player player) const {
    return Board::Pattern(
        pattern.positions &
        (~pieces[static_cast<std::size_t>(player)].positions));
  }

  /**
   * @param p The pattern to check.
   *
   * @return True if the all of the given positions are unoccupied by pieces of
   * either player.
   */
  bool contains_spaces(const Pattern p) const {
    return Pattern(pieces[static_cast<size_t>(Player::Player1)].positions |
                   pieces[static_cast<size_t>(Player::Player2)].positions)
               .count_overlap(p) == 0;
  }

  /**
   * @param m The move to check for.
   *
   * @return True if the board contains the given move.
   */
  bool contains_move(Move m) const {
    return pieces[static_cast<size_t>(m.player)].positions.test(
        m.board_position);
  }

  /**
   * @param position The position to check for a move.
   *
   * @return True if a move has been played by either player at the given
   * position.
   */
  bool contains_spaces(std::size_t position) const {
    return !(
        pieces[static_cast<size_t>(Player::Player1)].positions.test(position) ||
        pieces[static_cast<size_t>(Player::Player2)].positions.test(position));
  }

  /**
   * @param p The pattern to check.
   * @param player The player to check.
   *
   * @return True if the given player's moveset contains the given pattern.
   */
  bool contains(Pattern p, Player player) const {
    return pieces[static_cast<size_t>(player)].contains(p);
  }

  /**
   * @param m The move to add to the board.
   */
  void add(const Move m) {
    if (m.player != active_player()) {
      throw std::logic_error("Supplied move is not legal on the given board!");
    }

    if (!contains_spaces(m.board_position))
      throw std::invalid_argument("Piece already exists at position " +
                                  std::to_string(m.board_position));
    pieces[static_cast<size_t>(m.player)].positions.set(m.board_position);
  }

  /**
   * @param m The move to remove from the board.
   */
  void remove(const Move m) {
    if (m.player != get_other_player(active_player())) {
      throw std::logic_error(
          "Removing given move would lead to an illegal board state!");
    }

    if (!pieces[static_cast<size_t>(m.player)].positions.test(m.board_position))
      throw std::invalid_argument(
          "Piece does not exist at position " +
          std::to_string(m.board_position) + " for player " +
          std::to_string(static_cast<size_t>(m.player) + 1U));
    pieces[static_cast<size_t>(m.player)].positions.reset(m.board_position);
  }

  /**
   * Produces a new board with the given move added.
   *
   * @param m The move to add.
   *
   * @return The new board.
   */
  Board operator+(const Move m) const {
    Board temp(*this);
    temp.add(m);
    return temp;
  }

  /**
   * Produces a new board with the given move removed.
   *
   * @param m The move to add.
   *
   * @return The new board.
   */
  Board operator-(const Move m) const {
    Board temp(*this);
    temp.remove(m);
    return temp;
  }

  /**
   * @param b A different board.
   *
   * @return True if the given board is equivalent to this board. Note that the
   * histories of the two boards may be different.
   */
  bool operator==(const Board &b) const {
    return ((pieces[static_cast<size_t>(Player::Player2)] ==
             b.pieces[static_cast<size_t>(Player::Player2)]) &&
            (pieces[static_cast<size_t>(Player::Player1)] ==
             b.pieces[static_cast<size_t>(Player::Player1)]));
  }

  /**
   * @param b A different board.
   *
   * @return True if the given board is not equivalent to this board.
   */
  bool operator!=(const Board &b) const { return !(*this == b); }
};

}  // namespace NInARow

#endif  // NINAROW_BOARD_H_INCLUDED
