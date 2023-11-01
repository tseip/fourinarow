#ifndef NINAROW_PATTERN_H_INCLUDED
#define NINAROW_PATTERN_H_INCLUDED

#include <bitset>
#include <stdexcept>
#include <string>

#include "player.h"

namespace NInARow {

/**
 * Represents a set of positions on the game board.
 *
 * @tparam HEIGHT The height of the pattern.
 * @tparam WIDTH The width of the pattern.
 * @tparam N The number of stones in a row required to win.
 */
template <std::size_t HEIGHT, std::size_t WIDTH, std::size_t N>
class Pattern {
 private:
  static constexpr std::size_t BOARD_SIZE = HEIGHT * WIDTH;

 public:
  using bitset = std::bitset<BOARD_SIZE>;

  /**
   * Creates an empty pattern.
   */
  Pattern() = default;

  /**
   * Creates a pattern with the specified bitset.
   *
   * @param positions A number containing the bitset to be represented by this
   * pattern.
   */
  Pattern(unsigned long long positions) : Pattern(bitset{positions}) {}

  /**
   * Creates a pattern with the specified bitset.
   *
   * @param positions A bitset to be represented by this pattern.
   */
  Pattern(const bitset positions) : positions(positions) {}

  /**
   * Creates a pattern from a binary string.
   */
  Pattern(const std::string s) : positions(bitset(std::stoull(s, nullptr, 2))) {
    if (s.size() != BOARD_SIZE)
      throw std::invalid_argument(
          "Provided string incompatible length to uniquely specify pattern.");
  }

  /**
   * Turns a pattern into a binary string.
   */
  std::string to_string() const { return positions.to_string(); }

  /**
   * Shifts the pattern by row, col.
   *
   * @param row The number of rows to shift the pattern by.
   * @param col The number of columns to shift the pattern by.
   */
  void shift(int row, int col) {
    const static bitset all_but_last_col = []() {
      bitset last_col;
      for (size_t i = 0; i < HEIGHT; ++i) {
        last_col.set((i + 1U) * WIDTH - 1U);
      }
      return last_col.flip();
    }();
    const static bitset all_but_first_col = []() {
      bitset first_col;
      for (size_t i = 0; i < HEIGHT; ++i) {
        first_col.set(i * WIDTH);
      }
      return first_col.flip();
    }();
    if (col >= 0) {
      for (size_t i = 0; i < static_cast<std::size_t>(col); ++i) {
        positions &= all_but_last_col;
        positions <<= 1;
      }
    } else {
      for (size_t i = 0; i < static_cast<std::size_t>(-col); ++i) {
        positions &= all_but_first_col;
        positions >>= 1;
      }
    }
    if (row >= 0) {
      positions <<= static_cast<size_t>(row * WIDTH);
    } else {
      positions >>= static_cast<size_t>(-row * WIDTH);
    }
  }

  /**
   * @return The total maximum height of the pattern.
   */
  static std::size_t get_height() { return HEIGHT; }

  /**
   * @return The total maximum width of the pattern.
   */
  static std::size_t get_width() { return WIDTH; }

  /**
   * @return True if this pattern contains no set bits.
   */
  bool is_empty() const { return positions.none(); }

  /**
   * @return The index of the minimum row containing a set bit. If no bit is
   * set, return 0.
   */
  std::size_t min_row() const {
    if (is_empty()) return 0;
    Pattern temp(positions);
    std::size_t row = HEIGHT;
    while (temp.positions.any()) {
      temp.shift(1, 0);
      row--;
    }
    return row;
  }

  /**
   * @return The index of the maximum row containing a set bit. If no bit is
   * set, return total height.
   */
  std::size_t max_row() const {
    if (is_empty()) return HEIGHT;
    Pattern temp(positions);
    std::size_t row = 0;
    while (temp.positions.any()) {
      temp.shift(-1, 0);
      row++;
    }
    row -= 1U;
    return row;
  }

  /**
   * @return The index of the minimum column containing a set bit. If no bit is
   * set, return 0.
   */
  std::size_t min_col() const {
    if (is_empty()) return 0;
    Pattern temp(positions);
    std::size_t col = WIDTH;
    while (temp.positions.any()) {
      temp.shift(0, 1);
      col--;
    }
    return col;
  }

  /**
   * @return The index of the maximum column containing a set bit. If no bit is
   * set, return total width.
   */
  std::size_t max_col() const {
    if (is_empty()) return WIDTH;
    Pattern temp(positions);
    std::size_t col = 0;
    while (temp.positions.any()) {
      temp.shift(0, -1);
      col++;
    }
    col -= 1U;
    return col;
  }

  /**
   * @return True if a win exists in the pattern, false otherwise.
   */
  bool contains_win() const {
    /*
     * Check for a vertical win. Drag the pattern against itself vertically
     * for the length of the win window, and then check to see if any valid
     * starting locations are flagged.
     */
    const auto vertical_win = [&]() {
      static const bitset vertical_mask = []() {
        bitset vertical_mask;
        for (std::size_t row = 0; row < HEIGHT - (N - 1U); ++row) {
          for (std::size_t column = 0; column < WIDTH; ++column) {
            vertical_mask.set(row * WIDTH + column);
          }
        }
        return vertical_mask;
      }();

      bitset p = positions;
      for (size_t i = 0; i < N; ++i) {
        p &= positions >> i * WIDTH;
      }
      return (p & vertical_mask).any();
    };

    /*
     * Check for a vertical win. Drag the pattern against itself horizontally
     * for the length of the win window, and then check to see if any valid
     * starting locations are flagged.
     */
    const auto horizontal_win = [&]() {
      static const bitset horizontal_mask = []() {
        bitset horizontal_mask;
        for (std::size_t row = 0; row < HEIGHT; ++row) {
          for (std::size_t column = 0; column < WIDTH - (N - 1U); ++column) {
            horizontal_mask.set(row * WIDTH + column);
          }
        }
        return horizontal_mask;
      }();

      bitset p = positions;
      for (size_t i = 0; i < N; ++i) {
        p &= positions >> i;
      }
      return (p & horizontal_mask).any();
    };

    /*
     * Check for a left diagonal win. Drag the pattern against itself
     * diagonally down and to the left for the length of the win window, and
     * then check to see if any valid starting locations are flagged.
     */
    const auto left_diagonal_win = [&]() {
      static const bitset left_diagonal_mask = []() {
        bitset left_diagonal_mask;
        for (std::size_t row = 0; row < HEIGHT - (N - 1U); ++row) {
          for (std::size_t column = N - 1U; column < WIDTH; ++column) {
            left_diagonal_mask.set(row * WIDTH + column);
          }
        }
        return left_diagonal_mask;
      }();

      bitset p = positions;
      for (size_t i = 0; i < N; ++i) {
        p &= positions >> i * (WIDTH - 1);
      }
      return (p & left_diagonal_mask).any();
    };

    /*
     * Check for a right diagonal win. Drag the pattern against itself
     * diagonally down and to the right for the length of the win window, and
     * then check to see if any valid starting locations are flagged.
     */
    const auto right_diagonal_win = [&]() {
      static const bitset right_diagonal_mask = []() {
        bitset right_diagonal_mask;
        for (std::size_t row = 0; row < HEIGHT - (N - 1U); ++row) {
          for (std::size_t column = 0; column < WIDTH - (N - 1U); ++column) {
            right_diagonal_mask.set(row * WIDTH + column);
          }
        }
        return right_diagonal_mask;
      }();

      bitset p = positions;
      for (size_t i = 0; i < N; ++i) {
        p &= positions >> i * (WIDTH + 1);
      }
      return (p & right_diagonal_mask).any();
    };

    return vertical_win() || right_diagonal_win() || left_diagonal_win() ||
           horizontal_win();
  }

  /**
   * @param p The pattern to check.
   *
   * @return The number of positions that both the given pattern and this
   * pattern share.
   */
  std::size_t count_overlap(const Pattern p) const {
    return (p.positions & positions).count();
  }

  /**
   * @param p The pattern to check.
   *
   * @return The number of positions in the given pattern that are not covered
   * by this pattern.
   */
  std::size_t count_spaces(const Pattern p) const {
    return (p.positions & ~positions).count();
  }

  /**
   * @param p The pattern to check.
   *
   * @return True if the given pattern is fully contained within this pattern.
   * An empty pattern is contained in all patterns.
   */
  bool contains(const Pattern p) const {
    return (p.positions & ~positions).none();
  }

  /**
   * @param p A different pattern.
   *
   * @return True if the given pattern is identical to this one.
   */
  bool operator==(const Pattern &p) const { return positions == p.positions; }

  /**
   * @param p A different pattern.
   *
   * @return True if the given pattern is not identical to this one.
   */
  bool operator!=(const Pattern &p) const { return !(*this == p); }

  /**
   * @return A list of 1-hot positions, where each element in the list
   * represents a single set element of the original position.
   */
  std::vector<Pattern> get_all_positions() const {
    std::vector<Pattern> all_positions;
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
      if (positions.test(i)) {
        all_positions.emplace_back(1LLU << i);
      }
    }

    return all_positions;
  }

  /**
   * @return A list of indices referencing set bits in the given position.
   */
  std::vector<std::size_t> get_all_position_indices() const {
    std::vector<std::size_t> position_indices;
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
      if (positions.test(i)) {
        position_indices.emplace_back(i);
      }
    }

    return position_indices;
  }

  /**
   * Encodes positions on the game board into a bitset, where board positions
   * are specified starting at the LSB of the bitset for the upper left
   * position and incrementing along columns, e.g., for a 4x3 board, the bit
   * positions are as follows:
   *
   * ----------
   * | 0| 1| 2|
   * ----------
   * | 3| 4| 5|
   * ----------
   * | 6| 7| 8|
   * ----------
   * | 9|10|11|
   * ----------
   */
  bitset positions;
};

/**
 * A hash function, to allow us to use Patterns as keys in maps.
 */
template <typename Pattern>
struct PatternHasher {
  /**
   * @param k The pattern to hash.
   *
   * @return The hash of the pattern.
   */
  std::size_t operator()(const Pattern &k) const {
    return std::hash<typename Pattern::bitset>()(k.positions);
  }
};

}  // namespace NInARow

#endif  // NINAROW_PATTERN_H_INCLUDED
