#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

/**
 * Represents the two players in a board game.
 */
enum class Player { Player1 = 0, Player2 = 1 };

/**
 * @param A given player.
 *
 * @return The player who is not the given player.
 */
inline Player get_other_player(const Player player) {
  return player == Player::Player1 ? Player::Player2 : Player::Player1;
}

#endif  // PLAYER_H_INCLUDED
