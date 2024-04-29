#ifndef NINAROW_HEURISTIC_H_INCLUDED
#define NINAROW_HEURISTIC_H_INCLUDED

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>

#include "bfs_node.h"
#include "fourbynine_features.h"
#include "ninarow_board.h"
#include "ninarow_heuristic_feature.h"
#include "ninarow_vectorized_feature_evaluator.h"
#include "searches.h"

namespace NInARow {

/**
 * Stores the evaluation weights for a group of features.
 */
struct FeatureGroupWeight {
  /**
   * The weight given to the feature when it's being evaluated from the
   * perspective of the active player.
   */
  double weight_act;

  /**
   * The weight given to the feature when it's being evaluated from the
   * perspective of the passive player.
   */
  double weight_pass;

  /**
   * The percent chance that a feature in this group will be ignored randomly.
   * Ranges from 0 to 1.
   */
  double drop_rate;

  /**
   * Default constructor.
   */
  FeatureGroupWeight() : weight_act(0.0), weight_pass(0.0), drop_rate(0.0) {}

  /**
   * Constructor.
   *
   * @param weight_act The weight given to the feature when it's being evaluated
   * from the perspective of the active player.
   * @param weight_pass The weight given to the feature when it's being
   * evaluated from the perspective of the passive player.
   * @param drop_rate The percent chance that a feature in this group will be
   * ignored randomly. Ranges from 0 to 1.
   */
  FeatureGroupWeight(double weight_act, double weight_pass, double drop_rate)
      : weight_act(weight_act),
        weight_pass(weight_pass),
        drop_rate(drop_rate) {}

  /**
   * @return The difference between the active and passive weights for this
   * feature group.
   */
  double diff_act_pass() const { return weight_act - weight_pass; }
};

/**
 * A helper class that augments a given feature with metadata that the heuristic
 * needs to keep track of during execution.
 *
 * @tparam Board The class representing a board for this class.
 */
template <typename Board>
struct HeuristicFeatureWithMetadata {
  /**
   * The feature this class is wrapping.
   */
  HeuristicFeature<Board> feature;

  /**
   * The index of this feature in the heuristic's master feature list. Used for
   * fast lookups.
   */
  std::size_t vector_index;

  /**
   * The index of the weights for this feature in the heuristic's master weight
   * list. Also indicates the group that this feature belongs to.
   */
  std::size_t weight_index;

  /**
   * If true, this feature should be evaluated. Features are turned off with
   * `drop_rate` probability by the heuristic.
   */
  bool enabled;

  /**
   * Default constructor.
   */
  HeuristicFeatureWithMetadata() = default;

  /**
   * Constructor.
   *
   * @param feature The feature this class is wrapping.
   * @param vector_index The index of this feature in the heuristic's master
   * feature list.
   * @param weight_index The index of the weights for this feature in the
   * heuristic's master weight list.
   */
  HeuristicFeatureWithMetadata(const HeuristicFeature<Board>& feature,
                               std::size_t vector_index,
                               std::size_t weight_index)
      : feature(feature),
        vector_index(vector_index),
        weight_index(weight_index),
        enabled(true) {}
};

/**
 * A heuristic for games of n-in-a-row.
 *
 * @tparam Board The board representation used by this heuristic.
 */
template <typename Board>
class Heuristic : public std::enable_shared_from_this<Heuristic<Board>> {
 public:
  using Feature = HeuristicFeature<Board>;
  using BoardT = Board;

 private:
  /**
   * A parameter controlling when searches should stop executing. The stopping
   * threshold is the number of times that a given move needs to be evaluated by
   * a tree search as the best consecutively before we terminate the search and
   * return the given move.
   */
  double stopping_thresh;

  /**
   * A parameter controlling how the heuristic will prune the search tree. In
   * normal operation, the heuristic will evaluate all possible moves from a
   * given position and return a heuristic value for each. It will then return a
   * subset of those moves to be searched via any tree search algorithm
   * operating on top of it. It will prune all moves from the moveset that are
   * worse than the best move in a given position by `pruning_threshold`, as
   * determined by the heuristic evaluation function.
   */
  double pruning_thresh;

  /**
   * A parameter controlling when searches should stop executing. A search will
   * only execute for a certain number of maximum iterations given by a function
   * of `gamma`: a maximum of 1 + 1.0 / gamma iterations will be performed by
   * searches that respect `gamma`.
   */
  double gamma;

  /**
   * The percent chance that the heuristic will simply return a random move.
   * Represents a lapse of attention.
   */
  double lapse_rate;

  /**
   * A scaling factor for the feature weights that can be varied by the Bayesian
   * optimization process.
   */
  double opp_scale;

  /**
   * An parameter used exclusively for Monte Carlo search; currently unused in
   * this implementation.
   */
  double exploration_constant;

  /**
   * Both of these parameters are direct functions of `opp_scale`; see
   * `opp_scale`'s documentation.
   * @{
   */
  double c_self;
  double c_opp;
  /**
   * @}
   */

  /**
   * A parameter controlling how much the heuristic should prefer the center of
   * the board.
   */
  double center_weight;

  /**
   * Our internal random number generator.
   */
  std::mt19937_64 engine;

  /**
   * Holds a list of weights for all of the features of the heuristic.
   */
  std::vector<FeatureGroupWeight> feature_group_weights;

  /**
   * Holds all of the features of the heuristic.
   */
  std::vector<HeuristicFeatureWithMetadata<Board>> features;

  /**
   * A helper class used to evaluate all of the features on the board in
   * parallel quickly.
   */
  VectorizedFeatureEvaluator<Board> feature_evaluator;

  /**
   * A static weight given to each tile on the board as function of the tile's
   * position by the heuristic. Prefers the center of the board.
   */
  std::array<double, Board::get_board_size()> vtile;

  /**
   * A random distribution used for supplying the heuristic evaluation function
   * with a noise parameter.
   */
  std::normal_distribution<double> noise;

  /**
   * A random distribution used for determining when the heuristic evaluation
   * should lapse and choose a random move. See `lapse_rate`.
   */
  std::bernoulli_distribution lapse;

  /**
   * If true, noise is injected across the evaluation function, including random
   * feature dropout. If false, the heuristic will evaluate deterministically.
   */
  bool noise_enabled;

  /**
   * Some state to keep track of whether or not a search is currently being
   * executed. Used for inspecting the search during execution.
   */
  bool search_in_progress;

 public:
  /**
   * Creates a heuristic.
   *
   * @param params The parameters to use for the heuristic.
   * @param add_default_features If true, use the default feature set in
   * `fourbynine_features.h`. If false, don't inject any features.
   *
   * @return A pointer to a newly created heuristic.
   */
  static std::shared_ptr<Heuristic> create(
      const std::vector<double>& params = DefaultFourByNineParameters,
      bool add_default_features = true) {
    auto heuristic = std::shared_ptr<Heuristic>(new Heuristic(params));
    if (add_default_features) {
      for (size_t i = 0; i < FourByNineFeatures.size(); ++i) {
        for (auto& feature : FourByNineFeatures[i]) {
          heuristic->add_feature(i, feature);
        }
      }
    }
    return heuristic;
  }

 private:
  /**
   * Constructor.
   *
   * @param params The parameters for this heuristic.
   */
  Heuristic(const std::vector<double>& params)
      : engine(),
        feature_group_weights(),
        features(),
        feature_evaluator(),
        vtile(),
        noise(),
        lapse(),
        noise_enabled(true),
        search_in_progress(false) {
    if (params.size() < 7 || (params.size() - 7) % 3 != 0) {
      throw std::invalid_argument(
          "The incorrect number of parameters have been passed to the "
          "heuristic function.");
    }
    std::size_t i = 0;
    stopping_thresh = params[i++];
    pruning_thresh = params[i++];
    gamma = params[i++];
    lapse_rate = params[i++];
    opp_scale = params[i++];
    exploration_constant = params[i++];
    center_weight = params[i++];
    const std::size_t num_param_packs =
        static_cast<std::size_t>((params.size() - 7) / 3);
    const std::size_t param_pack_idx = i;
    for (std::size_t j = 0; j < num_param_packs; ++j) {
      add_feature_group(params[param_pack_idx + j],
                        params[param_pack_idx + j + num_param_packs],
                        params[param_pack_idx + j + 2 * num_param_packs]);
    }
    noise = std::normal_distribution<double>(0.0, 1.0);
    lapse = std::bernoulli_distribution(lapse_rate);
    for (std::size_t i = 0; i < Board::get_board_size(); ++i)
      vtile[i] = 1.0 / sqrt(pow(i / Board::get_board_width() - 1.5, 2) +
                            pow(i % Board::get_board_width() - 4.0, 2));
    c_self = 2.0 * opp_scale / (1.0 + opp_scale);
    c_opp = 2.0 / (1.0 + opp_scale);
  }

 public:
  /**
   * Sets the seed for the internal random number generator.
   *
   * @param seed The seed to use for the random number generator.
   */
  void seed_generator(uint64_t seed) { engine.seed(seed); }

  /**
   * @return A list of feature group weights.
   */
  std::vector<FeatureGroupWeight>& get_feature_group_weights() {
    return feature_group_weights;
  }

  /**
   * @return All of the features in the heuristic, along with their associated
   * metadata.
   */
  std::vector<HeuristicFeatureWithMetadata<Board>>&
  get_features_with_metadata() {
    return features;
  }

  /**
   * Adds a new (empty) feature group to the heuristic.
   *
   * @param weight_act The weight given to the feature when it's being evaluated
   * from the perspective of the active player.
   * @param weight_pass The weight given to the feature when it's being
   * evaluated from the perspective of the passive player.
   * @param drop_rate The percent chance that a feature in this group will be
   * ignored randomly. Ranges from 0 to 1.
   */
  void add_feature_group(double weight_act, double weight_pass,
                         double drop_rate) {
    feature_group_weights.emplace_back(weight_act, weight_pass, drop_rate);
  }

  /**
   * Adds a single feature to the given feature group.
   *
   * @param i The index of the group to add the feature to.
   * @param feature The feature to add to the group.
   */
  void add_feature(std::size_t i, const Feature& feature) {
    if (i >= feature_group_weights.size()) {
      throw std::out_of_range(
          "Trying to add a feature to a non-existent feature group.");
    }
    features.emplace_back(feature, feature_evaluator.register_feature(feature),
                          i);
  }

  /**
   * Evaluates a given board position and returns a heuristic value for it.
   *
   * @param b The board to evaluate.
   *
   * @return The value of the heuristic evaluation function of the given
   * position.
   */
  double evaluate(const Board& b) const {
    const Player player = b.active_player();
    const Player other_player = get_other_player(player);
    double val = 0.0;

    for (const auto i : b.get_pieces(player).get_all_position_indices()) {
      val += center_weight * vtile[i];
    }

    for (const auto i : b.get_pieces(other_player).get_all_position_indices()) {
      val -= center_weight * vtile[i];
    }

    const auto player_pieces = feature_evaluator.query_pieces(b, player);
    const auto opponent_pieces =
        feature_evaluator.query_pieces(b, other_player);
    const auto spaces = feature_evaluator.query_spaces(b);
    for (const auto& feature : features) {
      if (!feature.enabled) continue;
      const auto i = feature.vector_index;
      if (feature.feature.contained_in(player_pieces[i], spaces[i])) {
        val += feature_group_weights[feature.weight_index].weight_act;
      } else if (feature.feature.contained_in(opponent_pieces[i], spaces[i])) {
        val -= feature_group_weights[feature.weight_index].weight_pass;
      }
    }
    return player == Player::Player1 ? val : -val;
  }

  /**
   * Returns all possible moves from a given position, as well as their
   * associated heuristic evaluations.
   *
   * @param b The board containing the starting position.
   * @param evalPlayer The player from whose perspective we're evaluating the
   * board.
   * @param sorted If true, return all of the moves in sorted order by heuristic
   * evaluation.
   *
   * @return All possible moves from the given position, evaluated by the
   * heuristic.
   */
  std::vector<typename Board::MoveT> get_moves(const Board& b,
                                               Player evalPlayer,
                                               bool sorted = true) {
    const Player player = b.active_player();
    const Player other_player = get_other_player(player);
    const double c_act = (player == evalPlayer) ? c_self : c_opp;
    const double c_pass = (player == evalPlayer) ? c_opp : c_self;

    auto player_pieces = feature_evaluator.query_pieces(b, player);
    auto opponent_pieces = feature_evaluator.query_pieces(b, other_player);
    auto spaces = feature_evaluator.query_spaces(b);

    std::unordered_map<typename Board::PatternT, typename Board::MoveT,
                       typename Board::PatternHasherT>
        candidate_moves;
    double deltaL = 0.0;
    for (const auto& feature : features) {
      if (!feature.enabled) continue;
      const auto i = feature.vector_index;
      if (feature.feature.contained_in(player_pieces[i], spaces[i])) {
        deltaL -= c_pass *
                  feature_group_weights[feature.weight_index].diff_act_pass();
      } else if (feature.feature.contained_in(opponent_pieces[i], spaces[i])) {
        deltaL -=
            c_act * feature_group_weights[feature.weight_index].diff_act_pass();
      }
    }

    for (const auto i : b.get_spaces().get_all_position_indices()) {
      candidate_moves[typename Board::PatternT(1LLU << i)] =
          typename Board::MoveT(i,
                                deltaL + center_weight * vtile[i] +
                                    (noise_enabled ? noise(engine) : 0.0),
                                player);
    }

    for (const auto& feature : features) {
      if (!feature.enabled) continue;
      const auto i = feature.vector_index;

      // If either player can fill in the feature, and the current player
      // can complete it...
      if (feature.feature.can_be_completed(player_pieces[i], opponent_pieces[i],
                                           spaces[i])) {
        const typename Board::PatternT player_missing_pieces =
            feature.feature.missing_pieces(b, player);
        auto search = candidate_moves.find(player_missing_pieces);
        if (search != candidate_moves.end()) {
          search->second.val +=
              c_pass * feature_group_weights[feature.weight_index].weight_pass;
        }
      }

      // If the current player has the required pieces but the opponent can
      // block us or if the other player has the feature and we can block
      // them...
      const bool can_be_removed =
          feature.feature.can_be_removed(player_pieces[i], spaces[i]);
      const bool can_remove_opponent =
          feature.feature.can_be_removed(opponent_pieces[i], spaces[i]);
      if (can_be_removed || can_remove_opponent) {
        for (const auto& position :
             feature.feature.spaces.get_all_positions()) {
          if (b.contains_spaces(position)) {
            auto search = candidate_moves.find(position);
            if (search != candidate_moves.end()) {
              if (can_be_removed)
                search->second.val -=
                    c_pass *
                    feature_group_weights[feature.weight_index].weight_pass;
              if (can_remove_opponent)
                search->second.val +=
                    c_act *
                    feature_group_weights[feature.weight_index].weight_act;
            }
          }
        }
      }
    }

    std::vector<typename Board::MoveT> output_moves;
    for (const auto kv : candidate_moves) {
      output_moves.push_back(kv.second);
    }
    std::sort(output_moves.begin(), output_moves.end(),
              [](const auto& m1, const auto& m2) {
                return m1.board_position < m2.board_position;
              });

    if (!sorted) return output_moves;
    std::sort(output_moves.begin(), output_moves.end(), std::greater<>());
    return output_moves;
  }

  /**
   * Returns a pruned set of moves from the given position. Evaluates every
   * move, and then removes the weakest moves as determined by `pruning_thresh`.
   *
   * @param b The board containing the starting position.
   * @param evalPlayer The player from whose perspective we're evaluating the
   * board.
   *
   * @return Pruned moves from the given position, evaluated by the heuristic.
   */
  std::vector<typename Board::MoveT> get_pruned_moves(const Board& b,
                                                      Player evalPlayer) {
    std::vector<typename Board::MoveT> candidates = get_moves(b, evalPlayer);
    std::size_t i = 1;
    while (i < candidates.size() &&
           abs(candidates[0].val - candidates[i].val) < pruning_thresh) {
      ++i;
    }
    if (i < candidates.size())
      candidates.erase(candidates.begin() + i, candidates.end());
    return candidates;
  }

  /**
   * @param b The board containing the starting position.
   *
   * @return A legal move selected uniformly at random from all possible moves
   * on the board.
   */
  typename Board::MoveT get_random_move(const Board& b) {
    std::vector<std::size_t> options;

    for (const auto i : b.get_spaces().get_all_position_indices()) {
      options.push_back(i);
    }

    if (options.size() > 0) {
      return typename Board::MoveT(options[std::uniform_int_distribution<int>(
                                       0, options.size() - 1U)(engine)],
                                   0.0, b.active_player());
    } else {
      return typename Board::MoveT(0, 0.0, b.active_player());
    }
  }

  /**
   * A helper function for allowing the heuristic to apply its lapse rate to a
   * given move selection. Does not perform any evaluation itself - assumes that
   * the given tree has already been built up by a search and returns either the
   * best move in the tree, or a lapsed random move if the lapse rate determines
   * we should.
   *
   * @param tree The tree of moves over which we'd like to select the best.
   *
   * @return Either the best move in the given tree, or a random move if we
   * lapse.
   */
  typename Board::MoveT get_best_move(std::shared_ptr<Node<Board>> tree) {
    if (noise_enabled && lapse(engine))
      return get_random_move(tree->get_board());

    return tree->get_best_move();
  }

  /**
   * Tells the heuristic that a search is in progress, triggering the removal
   * of a random subset of features as determined by their respective
   * `drop_rate`s.
   */
  void start_search() {
    if (search_in_progress)
      throw std::logic_error(
          "Cannot start a search when a previous search is being executed!");
    search_in_progress = true;
    if (noise_enabled) remove_features();
  }

  /**
   * Tells the heuristic that a search has completed and restores all dropped
   * features.
   */
  void complete_search() {
    restore_features();
    search_in_progress = false;
  }

  /**
   * @param enabled If true, enable noise, else, disable noise.
   */
  void set_noise_enabled(bool enabled) { noise_enabled = enabled; }

  /**
   * @return The `gamma` parameter.
   */
  double get_gamma() const { return gamma; }

  /**
   * @return The `stopping_thresh` parameter.
   */
  double get_stopping_thresh() const { return stopping_thresh; }

 private:
  /**
   * Randomly removes features, respecting their associated `drop_rate`s.
   */
  void remove_features() {
    for (auto& feature : features) {
      if (std::bernoulli_distribution{
              feature_group_weights[feature.weight_index].drop_rate}(engine)) {
        feature.enabled = false;
      } else {
        feature.enabled = true;
      }
    }
  }

  /**
   * Restores all features to the pool.
   */
  void restore_features() {
    for (auto& feature : features) {
      feature.enabled = true;
    }
  }
};

}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_H_INCLUDED
