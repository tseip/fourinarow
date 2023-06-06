#include <gtest/gtest.h>

#include "ninarow_pattern.h"

using namespace NInARow;

/**
 * Tests Pattern from/to string implementation.
 */
TEST(NInARowPatternTest, TestFromString) {
  {
    const std::string board =
        "100111111"
        "011010000"
        "011010000"
        "100110000";
    {
      Pattern<4, 9, 4> pattern(board);
      EXPECT_EQ(board, pattern.to_string());
    }

    {
      // Wrong length.
      EXPECT_THROW((Pattern<3, 3, 3>(board)), std::invalid_argument);
    }
  }

  {
    const std::string board =
        "100"
        "010"
        "001";
    {
      Pattern<3, 3, 3> pattern(board);
      EXPECT_EQ(board, pattern.to_string());
    }
    {
      // Wrong length.
      EXPECT_THROW((Pattern<4, 9, 4>(board)), std::invalid_argument);
    }
  }
}

/**
 * Tests Pattern shift logic.
 */
TEST(NInARowPatternTest, TestShift) {
  using Pattern = Pattern<3, 3, 3>;
  const std::string board =
      "101"
      "010"
      "101";
  Pattern pattern(board);

  // Horizontal shifts.
  {
    Pattern temp(pattern);
    temp.shift(0, 1);
    EXPECT_EQ(
        "010"
        "100"
        "010",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(0, 2);
    EXPECT_EQ(
        "100"
        "000"
        "100",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(0, 3);
    EXPECT_EQ(
        "000"
        "000"
        "000",
        temp.to_string());
  }

  {
    Pattern temp(pattern);
    temp.shift(0, -1);
    EXPECT_EQ(
        "010"
        "001"
        "010",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(0, -2);
    EXPECT_EQ(
        "001"
        "000"
        "001",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(0, -3);
    EXPECT_EQ(
        "000"
        "000"
        "000",
        temp.to_string());
  }

  // Vertical shifts.
  {
    Pattern temp(pattern);
    temp.shift(1, 0);
    EXPECT_EQ(
        "010"
        "101"
        "000",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(2, 0);
    EXPECT_EQ(
        "101"
        "000"
        "000",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(3, 0);
    EXPECT_EQ(
        "000"
        "000"
        "000",
        temp.to_string());
  }

  {
    Pattern temp(pattern);
    temp.shift(-1, 0);
    EXPECT_EQ(
        "000"
        "101"
        "010",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(-2, 0);
    EXPECT_EQ(
        "000"
        "000"
        "101",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(-3, 0);
    EXPECT_EQ(
        "000"
        "000"
        "000",
        temp.to_string());
  }

  // Diagonal shifts.
  {
    Pattern temp(pattern);
    temp.shift(1, 1);
    EXPECT_EQ(
        "100"
        "010"
        "000",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(2, 2);
    EXPECT_EQ(
        "100"
        "000"
        "000",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(3, 3);
    EXPECT_EQ(
        "000"
        "000"
        "000",
        temp.to_string());
  }
  {
    Pattern temp(pattern);
    temp.shift(-1, -2);
    EXPECT_EQ(
        "000"
        "001"
        "000",
        temp.to_string());
  }
}

/**
 * Tests Pattern min/max row/col logic.
 */
TEST(NInARowPatternTest, TestRowCol) {
  {
    const std::string board =
        "000"
        "010"
        "000";
    Pattern<3, 3, 3> pattern(board);
    std::size_t temp;
    EXPECT_TRUE(pattern.max_row(temp));
    EXPECT_EQ(temp, 1);
    EXPECT_TRUE(pattern.min_row(temp));
    EXPECT_EQ(temp, 1);
    EXPECT_TRUE(pattern.max_col(temp));
    EXPECT_EQ(temp, 1);
    EXPECT_TRUE(pattern.min_col(temp));
    EXPECT_EQ(temp, 1);
  }

  {
    const std::string board =
        "000"
        "000"
        "000";
    Pattern<3, 3, 3> pattern(board);
    std::size_t temp;
    EXPECT_FALSE(pattern.max_row(temp));
    EXPECT_FALSE(pattern.min_row(temp));
    EXPECT_FALSE(pattern.max_col(temp));
    EXPECT_FALSE(pattern.min_col(temp));
  }

  {
    const std::string board =
        "00001"
        "00000"
        "00000"
        "01000"
        "00000";
    Pattern<5, 5, 3> pattern(board);
    // Keep in mind, these strings are read in LSB to MSB,
    // so they are functionally mirrored diagonally.
    std::size_t temp;
    EXPECT_TRUE(pattern.max_row(temp));
    EXPECT_EQ(temp, 4);
    EXPECT_TRUE(pattern.min_row(temp));
    EXPECT_EQ(temp, 1);
    EXPECT_TRUE(pattern.max_col(temp));
    EXPECT_EQ(temp, 3);
    EXPECT_TRUE(pattern.min_col(temp));
    EXPECT_EQ(temp, 0);
  }
}

/**
 * Tests Pattern win detection logic.
 */
TEST(NInARowPatternTest, TestContainsWin) {
  {
    std::vector<std::pair<std::string, bool>> testcases = {{"000"
                                                            "000"
                                                            "000",
                                                            false},
                                                           {"110"
                                                            "110"
                                                            "000",
                                                            false},
                                                           {"000"
                                                            "011"
                                                            "011",
                                                            false},
                                                           {"010"
                                                            "101"
                                                            "010",
                                                            false},
                                                           {"101"
                                                            "101"
                                                            "010",
                                                            false},
                                                           {"010"
                                                            "101"
                                                            "101",
                                                            false},
                                                           {"111"
                                                            "111"
                                                            "111",
                                                            true},
                                                           {"111"
                                                            "000"
                                                            "000",
                                                            true},
                                                           {"100"
                                                            "100"
                                                            "100",
                                                            true},
                                                           {"100"
                                                            "010"
                                                            "001",
                                                            true},
                                                           {"001"
                                                            "010"
                                                            "100",
                                                            true},
                                                           {"111"
                                                            "101"
                                                            "111",
                                                            true},
                                                           {"110"
                                                            "011"
                                                            "011",
                                                            true}};
    for (auto &pair : testcases) {
      Pattern<3, 3, 3> pattern(pair.first);
      EXPECT_EQ(pattern.contains_win(), pair.second);
    }
  }

  {
    std::vector<std::pair<std::string, bool>> testcases = {{"100111101"
                                                            "011010000"
                                                            "011010000"
                                                            "100110000",
                                                            true},
                                                           {"000000000"
                                                            "001111000"
                                                            "000000000"
                                                            "000000000",
                                                            true},
                                                           {"001000000"
                                                            "001000000"
                                                            "001000000"
                                                            "001000000",
                                                            true},
                                                           {"010000000"
                                                            "001000000"
                                                            "000100000"
                                                            "000010000",
                                                            true},
                                                           {"000000001"
                                                            "000000010"
                                                            "000000100"
                                                            "000001000",
                                                            true},
                                                           {"101110000"
                                                            "010010001"
                                                            "001010010"
                                                            "000000100",
                                                            false}

    };
    for (auto &pair : testcases) {
      Pattern<4, 9, 4> pattern(pair.first);
      EXPECT_EQ(pattern.contains_win(), pair.second);
    }
    // All of the above patterns have a 3-win.
    for (auto &pair : testcases) {
      Pattern<4, 9, 3> pattern(pair.first);
      EXPECT_EQ(pattern.contains_win(), true);
    }
    // None have a 5-win.
    for (auto &pair : testcases) {
      Pattern<4, 9, 5> pattern(pair.first);
      EXPECT_EQ(pattern.contains_win(), false);
    }
  }

  {
    std::vector<std::pair<std::string, bool>> testcases = {{"000000"
                                                            "000000"
                                                            "001000"
                                                            "000100"
                                                            "000010"
                                                            "000001",
                                                            true},
                                                           {"000000"
                                                            "000000"
                                                            "000001"
                                                            "000010"
                                                            "000100"
                                                            "001000",
                                                            true},
                                                           {"000000"
                                                            "000100"
                                                            "001000"
                                                            "010000"
                                                            "100000"
                                                            "000000",
                                                            true},
                                                           {"001000"
                                                            "000100"
                                                            "000010"
                                                            "000001"
                                                            "000000"
                                                            "000000",
                                                            true},
                                                           {"100000"
                                                            "010000"
                                                            "001000"
                                                            "000000"
                                                            "000010"
                                                            "000001",
                                                            false},

                                                           {"000001"
                                                            "000010"
                                                            "000100"
                                                            "000000"
                                                            "010000"
                                                            "100000",
                                                            false}

    };
    for (auto &pair : testcases) {
      Pattern<6, 6, 4> pattern(pair.first);
      EXPECT_EQ(pattern.contains_win(), pair.second);
    }
  }
}

/**
 * Tests Pattern comparison logic.
 */
TEST(NInARowPatternTest, TestComparison) {
  using Pattern = Pattern<3, 3, 3>;
  Pattern test_pattern(
      "101"
      "010"
      "101");
  ASSERT_EQ(test_pattern.count_overlap(test_pattern),
            test_pattern.positions.count());

  ASSERT_EQ(test_pattern.count_spaces(test_pattern), 0U);

  ASSERT_TRUE(test_pattern.contains(test_pattern));

  Pattern all("111111111");
  ASSERT_EQ(test_pattern.count_overlap(all), test_pattern.positions.count());
  ASSERT_EQ(all.count_overlap(test_pattern), test_pattern.positions.count());

  ASSERT_EQ(test_pattern.count_spaces(all), 4U);
  ASSERT_EQ(all.count_spaces(test_pattern), 0U);

  ASSERT_FALSE(test_pattern.contains(all));
  ASSERT_TRUE(all.contains(test_pattern));

  Pattern none;
  ASSERT_EQ(test_pattern.count_overlap(none), 0U);
  ASSERT_EQ(none.count_overlap(test_pattern), 0U);

  ASSERT_EQ(test_pattern.count_spaces(none), 0U);
  ASSERT_EQ(none.count_spaces(test_pattern), test_pattern.positions.count());

  ASSERT_TRUE(test_pattern.contains(none));
  ASSERT_FALSE(none.contains(test_pattern));

  Pattern anti_test_pattern(~test_pattern.positions);
  ASSERT_EQ(test_pattern.count_overlap(anti_test_pattern), 0U);
  ASSERT_EQ(anti_test_pattern.count_overlap(test_pattern), 0U);

  ASSERT_EQ(test_pattern.count_spaces(anti_test_pattern),
            anti_test_pattern.positions.count());
  ASSERT_EQ(anti_test_pattern.count_spaces(test_pattern),
            test_pattern.positions.count());

  ASSERT_FALSE(test_pattern.contains(anti_test_pattern));
  ASSERT_FALSE(anti_test_pattern.contains(test_pattern));

  Pattern sub_pattern(
      "100"
      "010"
      "001");
  ASSERT_EQ(test_pattern.count_overlap(sub_pattern), 3U);
  ASSERT_EQ(sub_pattern.count_overlap(test_pattern), 3U);

  ASSERT_EQ(test_pattern.count_spaces(sub_pattern), 0U);
  ASSERT_EQ(sub_pattern.count_spaces(test_pattern), 2U);

  ASSERT_TRUE(test_pattern.contains(sub_pattern));
  ASSERT_FALSE(sub_pattern.contains(test_pattern));
}
