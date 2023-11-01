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

struct FeatureGroupWeight {
  double weight_act;
  double weight_pass;
  double drop_rate;

  FeatureGroupWeight() = default;

  FeatureGroupWeight(double w_act, double w_pass, double drop_rate)
      : weight_act(w_act), weight_pass(w_pass), drop_rate(drop_rate) {}

  double diff_act_pass() const { return weight_act - weight_pass; }
};

template <typename Board>
struct HeuristicFeatureWithMetadata {
  HeuristicFeature<Board> feature;
  std::size_t vector_index;
  std::size_t weight_index;
  bool enabled;
  HeuristicFeatureWithMetadata() = default;
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
 */
template <typename Board>
class Heuristic : public std::enable_shared_from_this<Heuristic<Board>> {
 public:
  using Feature = HeuristicFeature<Board>;
  using BoardT = Board;

 private:
  double stopping_thresh;
  double pruning_thresh;
  double gamma;
  double lapse_rate;
  double opp_scale;
  double exploration_constant;
  double c_self;
  double c_opp;
  double center_weight;
  std::mt19937_64 engine;
  std::vector<FeatureGroupWeight> feature_group_weights;
  std::vector<HeuristicFeatureWithMetadata<Board>> features;
  VectorizedFeatureEvaluator<Board> feature_evaluator;
  std::array<double, Board::get_board_size()> vtile;
  std::normal_distribution<double> noise;
  std::bernoulli_distribution lapse;
  bool noise_enabled;
  bool search_in_progress;

 public:
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
    get_params_from_vector(params);
    update();
  }

  void get_params_from_vector(const std::vector<double>& params) {
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
  }

  void update() {
    noise = std::normal_distribution<double>(0.0, 1.0);
    lapse = std::bernoulli_distribution(lapse_rate);
    for (std::size_t i = 0; i < Board::get_board_size(); ++i)
      vtile[i] = 1.0 / sqrt(pow(i / Board::get_board_width() - 1.5, 2) +
                            pow(i % Board::get_board_width() - 4.0, 2));
    c_self = 2.0 * opp_scale / (1.0 + opp_scale);
    c_opp = 2.0 / (1.0 + opp_scale);
  }

 public:
  void seed_generator(uint64_t seed) { engine.seed(seed); }

  std::vector<FeatureGroupWeight>& get_feature_group_weights() {
    return feature_group_weights;
  }

  std::vector<HeuristicFeatureWithMetadata<Board>>&
  get_features_with_metadata() {
    return features;
  }

  void add_feature_group(double w_act, double w_pass, double delta) {
    feature_group_weights.emplace_back(w_act, w_pass, delta);
  }

  void add_feature(std::size_t i, const Feature& feature) {
    if (i >= feature_group_weights.size()) {
      throw std::out_of_range(
          "Trying to add a feature to a non-existent feature group.");
    }
    features.emplace_back(feature, feature_evaluator.register_feature(feature),
                          i);
  }

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

  typename Board::MoveT get_best_move(
      const Board& b, Player player,
      std::shared_ptr<Search<Heuristic>> search) {
    if (search_in_progress)
      throw std::logic_error(
          "Cannot start a search when a previous search is being executed!");
    if (noise_enabled && lapse(engine)) return get_random_move(b);

    search->search(this->shared_from_this(), player, b);

    return search->get_tree()->get_best_move();
  }

  typename Board::MoveT get_best_known_move_from_search_tree(
      std::shared_ptr<Search<Heuristic>> search) {
    auto root = search->get_tree();
    if (noise_enabled && lapse(engine))
      return get_random_move(root->get_board());

    return root->get_best_move();
  }

  void start_search() {
    if (search_in_progress)
      throw std::logic_error(
          "Cannot start a search when a previous search is being executed!");
    search_in_progress = true;
    if (noise_enabled) remove_features();
  }

  void complete_search() {
    restore_features();
    search_in_progress = false;
  }

  void set_noise_enabled(bool enabled) { noise_enabled = enabled; }

  double get_gamma() const { return gamma; }

  double get_stopping_thresh() const { return stopping_thresh; }

 private:
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

  void restore_features() {
    for (auto& feature : features) {
      feature.enabled = true;
    }
  }
};

}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_H_INCLUDED
