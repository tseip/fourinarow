#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

/**
 * Represents the two players in a board game.
 */
enum class Player { Player1 = 0, Player2 = 1 };

/**
 * @param player A given player.
 *
 * @return The player who is not the given player.
 */
inline Player get_other_player(const Player player) {
  return player == Player::Player1 ? Player::Player2 : Player::Player1;
}

/**
 * @param player The player to convert to a bool.
 *
 * @return True if the player is Player2, else false.
 */
inline bool player_to_bool(const Player player) {
  return player == Player::Player2;
}

/**
 * @param player The bool to convert to a player.
 *
 * @return Player2 if the bool is true, else Player1.
 */
inline Player bool_to_player(const bool player) {
  return player ? Player::Player2 : Player::Player1;
}

/**
 * @param player The player to convert to a string representation.
 *
 * @return The given player as a string.
 */
inline std::string player_to_string(const Player player) {
  return player == Player::Player1 ? "Black" : "White";
}

#endif  // PLAYER_H_INCLUDED
