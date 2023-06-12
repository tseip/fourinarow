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
#include "searches.h"

namespace NInARow {

template <typename Board>
class FeaturePack {
 public:
  using Feature = HeuristicFeature<Board>;
  double weight_act;
  double weight_pass;
  double drop_rate;
  std::vector<Feature> features;

  FeaturePack() = default;

  FeaturePack(double w_act, double w_pass, double drop_rate)
      : weight_act(w_act), weight_pass(w_pass), drop_rate(drop_rate) {}

  double diff_act_pass() const { return weight_act - weight_pass; }
};

/**
 * A heuristic for games of n-in-a-row.
 */
template <typename Board>
class Heuristic : public std::enable_shared_from_this<Heuristic<Board>> {
 public:
  using Feature = HeuristicFeature<Board>;
  using FeatureP = FeaturePack<Board>;
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
  std::vector<FeatureP> feature_packs;
  std::array<double, Board::get_board_size()> vtile;
  std::normal_distribution<double> noise;
  std::bernoulli_distribution lapse;
  bool noise_enabled;
  bool search_in_progress;

  std::vector<std::vector<Feature>> removed_features;

 public:
  static std::shared_ptr<Heuristic> create(const std::vector<double>& params,
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
        feature_packs(),
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
      add_feature_pack(params[param_pack_idx + j],
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

  std::vector<FeatureP>& get_feature_packs() { return feature_packs; }

  void add_feature_pack(double w_act, double w_pass, double delta) {
    feature_packs.emplace_back(w_act, w_pass, delta);
  }

  void add_feature(std::size_t i, const Feature& feature) {
    if (i >= feature_packs.size()) {
      throw std::out_of_range(
          "Trying to add a feature to a non-existent feature pack.");
    }
    feature_packs[i].features.push_back(feature);
  }

  double evaluate(const Board& b) const {
    const Player player = b.active_player();
    double val = 0.0;
    for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
      const typename Board::PatternT pattern{1LLU << i};
      if (b.contains(pattern, player)) val += center_weight * vtile[i];
      if (b.contains(pattern, get_other_player(player)))
        val -= center_weight * vtile[i];
    }
    for (const auto& feature_pack : feature_packs) {
      for (const auto& feature : feature_pack.features) {
        if (feature.is_active(b, player) && feature.contained(b, player)) {
          val += feature_pack.weight_act;
        } else if (feature.is_active(b, get_other_player(player)) &&
                   feature.contained(b, get_other_player(player))) {
          val -= feature_pack.weight_pass;
        }
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

    std::unordered_map<typename Board::PatternT, typename Board::MoveT,
                       typename Board::PatternHasherT>
        candidate_moves;
    double deltaL = 0.0;
    for (const auto& feature_pack : feature_packs) {
      for (const auto& feature : feature_pack.features) {
        if (feature.contained(b, player) && feature.is_active(b, player))
          deltaL -= c_pass * feature_pack.diff_act_pass();
        else if (feature.contained(b, other_player) &&
                 feature.is_active(b, other_player))
          deltaL -= c_act * feature_pack.diff_act_pass();
      }
    }

    for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
      const std::size_t bit_position = 1LLU << i;
      const typename Board::PatternT pattern{bit_position};
      if (b.contains_spaces(pattern)) {
        candidate_moves[pattern] =
            typename Board::MoveT(i,
                                  deltaL + center_weight * vtile[i] +
                                      (noise_enabled ? noise(engine) : 0.0),
                                  player);
      }
    }
    for (const auto& feature_pack : feature_packs) {
      for (const auto& feature : feature_pack.features) {
        if (feature.is_active(b, player) ||
            feature.is_active(b, other_player)) {
          const typename Board::PatternT player_missing_pieces =
              feature.missing_pieces(b, player);
          const typename Board::PatternT other_player_missing_pieces =
              feature.missing_pieces(b, other_player);
          if ((player_missing_pieces.count_overlap(
                   other_player_missing_pieces) != 0) &&
              player_missing_pieces.positions.count() == 1) {
            auto search = candidate_moves.find(player_missing_pieces);
            if (search != candidate_moves.end()) {
              search->second.val += c_pass * feature_pack.weight_pass;
            }
          }
          if (player_missing_pieces == typename Board::PatternT{0} &&
              feature.just_active(b, player)) {
            for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
              const typename Board::PatternT position{1LLU << i};
              if (b.contains_spaces(position) &&
                  feature.contains_spaces(position)) {
                auto search = candidate_moves.find(position);
                if (search != candidate_moves.end()) {
                  search->second.val -= c_pass * feature_pack.weight_pass;
                }
              }
            }
          }
          if (other_player_missing_pieces == typename Board::PatternT{0} &&
              feature.just_active(b, other_player)) {
            for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
              const typename Board::PatternT position{1LLU << i};
              if (b.contains_spaces(position) &&
                  feature.contains_spaces(position)) {
                auto search = candidate_moves.find(position);
                if (search != candidate_moves.end()) {
                  search->second.val += c_act * feature_pack.weight_act;
                }
              }
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
    for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
      const typename Board::PatternT position{1LLU << i};
      if (b.contains_spaces(position)) {
        options.push_back(i);
      }
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
    for (std::size_t i = 0; i < feature_packs.size(); ++i) {
      removed_features.push_back({});
      for (auto it = feature_packs[i].features.begin();
           it != feature_packs[i].features.end();) {
        if (std::bernoulli_distribution{feature_packs[i].drop_rate}(engine)) {
          removed_features[i].push_back(*it);
          feature_packs[i].features.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  void restore_features() {
    for (std::size_t i = 0; i < removed_features.size(); ++i) {
      for (const auto& feature : removed_features[i]) {
        feature_packs[i].features.push_back(feature);
      }
      removed_features[i].clear();
    }
    removed_features.clear();
  }
};

static std::shared_ptr<Heuristic<Board<4, 9, 4>>> getDefaultFourByNineHeuristic(
    const std::vector<double>& parameters = DefaultFourByNineParameters) {
  auto heuristic = Heuristic<Board<4, 9, 4>>::create(parameters);
  for (size_t i = 0; i < FourByNineFeatures.size(); ++i) {
    for (auto& feature : FourByNineFeatures[i]) {
      heuristic->add_feature(i, feature);
    }
  }
  return heuristic;
}

}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_H_INCLUDED
