#ifndef NINAROW_MOVE_H_INCLUDED
#define NINAROW_MOVE_H_INCLUDED

#include <string>

#include "ninarow_pattern.h"
#include "player.h"

namespace NInARow {

/**
 * Represents a single move in a game of N in a Row.
 */
template <std::size_t HEIGHT, std::size_t WIDTH, std::size_t N>
class Move {
 private:
  using PatternT = Pattern<HEIGHT, WIDTH, N>;

 public:
  /**
   * The position of the move on the board. 0 is the upper left tile on the
   * board, and positions increment along columns.
   */
  std::size_t board_position;

  /**
   * The heuristic value of a given move.
   */
  double val;

  /**
   * The player who played the move.
   */
  Player player;

  /**
   * Default constructor.
   */
  Move() : Move(0, 0.0, Player::Player1){};

  /**
   * @param m The board position of the move, where m is the bit position
   *          of the move (m = 0 is the LSB).
   * @param v The heuristic value of the given move.
   * @param p The player who played the given move.
   *
   * @return A move played by player p at position [row, col] with value v.
   */
  Move(std::size_t m, double v, Player p)
      : board_position(m), val(v), player(p) {
    // Throw if position is out of bounds.
    PatternT pattern;
    (void)pattern.positions.test(board_position);
  }

  /**
   * @param row The row index of the given move.
   * @param col The column index of the given move.
   * @param v The heuristic value of the given move.
   * @param p The player who played the given move.
   *
   * @return A move played by player p at position [row, col] with value v.
   */
  Move(std::size_t row, std::size_t col, double v, Player p)
      : Move(row * WIDTH + col, v, p) {}

  /**
   * Compares two moves using the heuristic value.
   *
   * @param m The move to compare.
   *
   * @return True if the heuristic value of m is less than our heuristic
   * value.
   */
  bool operator<(const Move &m) const { return val < m.val; }

  /**
   * Compares two moves using the heuristic value.
   *
   * @param m The move to compare.
   *
   * @return True if the heuristic value of m is greater than our heuristic
   * value.
   */
  bool operator>(const Move &m) const { return val > m.val; }

  /**
   * @return The index of the row of this move.
   */
  std::size_t get_row() const { return board_position / WIDTH; }

  /**
   * @return The index of the column of this move.
   */
  std::size_t get_col() const { return board_position % WIDTH; }
};
}  // namespace NInARow

#endif  // NINAROW_MOVE_H_INCLUDED
