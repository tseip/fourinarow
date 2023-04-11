#ifndef NINAROW_HEURISTIC_H_INCLUDED
#define NINAROW_HEURISTIC_H_INCLUDED

#include <algorithm>
#include <array>
#include <fstream>
#include <random>
#include <unordered_map>

#include "game_tree.h"
#include "ninarow_board.h"
#include "ninarow_heuristic_feature.h"

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
class Heuristic {
 public:
  using Feature = HeuristicFeature<Board>;
  using FeatureP = FeaturePack<Board>;

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

 public:
  Heuristic(const std::vector<double>& params)
      : engine(),
        feature_packs(),
        vtile(),
        noise(),
        lapse(),
        noise_enabled(true) {
    get_params_from_vector(params);
    update();
  }

 private:
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
    std::size_t param_pack_idx = 0;
    for (std::size_t j = 0; j < num_param_packs; ++j) {
      add_feature_pack(params[3 * j], params[3 * j + 1], params[3 * j + 2]);
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
      const typename Board::Pattern pattern{1LLU << i};
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

  std::vector<typename Board::Move> get_moves(const Board& b, Player evalPlayer,
                                              bool sorted = true) {
    const Player player = b.active_player();
    const Player other_player = get_other_player(player);
    const double c_act = (player == evalPlayer) ? c_self : c_opp;
    const double c_pass = (player == evalPlayer) ? c_opp : c_self;

    std::unordered_map<typename Board::Pattern, typename Board::Move,
                       typename Board::PatternHasher>
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
      const typename Board::Pattern pattern{bit_position};
      if (b.contains_spaces(pattern)) {
        candidate_moves[pattern] =
            typename Board::Move(i,
                                 deltaL + center_weight * vtile[i] +
                                     (noise_enabled ? noise(engine) : 0.0),
                                 player);
      }
    }
    for (const auto& feature_pack : feature_packs) {
      for (const auto& feature : feature_pack.features) {
        if (feature.is_active(b, player) ||
            feature.is_active(b, other_player)) {
          const typename Board::Pattern player_missing_pieces =
              feature.missing_pieces(b, player);
          const typename Board::Pattern other_player_missing_pieces =
              feature.missing_pieces(b, other_player);
          if ((player_missing_pieces.count_overlap(
                   other_player_missing_pieces) != 0) &&
              player_missing_pieces.positions.count() == 1) {
            auto search = candidate_moves.find(player_missing_pieces);
            if (search != candidate_moves.end()) {
              search->second.val += c_pass * feature_pack.weight_pass;
            }
          }
          if (player_missing_pieces == typename Board::Pattern{0} &&
              feature.just_active(b, player)) {
            for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
              const typename Board::Pattern position{1LLU << i};
              if (b.contains_spaces(position) &&
                  feature.contains_spaces(position)) {
                auto search = candidate_moves.find(position);
                if (search != candidate_moves.end()) {
                  search->second.val -= c_pass * feature_pack.weight_pass;
                }
              }
            }
          }
          if (other_player_missing_pieces == typename Board::Pattern{0} &&
              feature.just_active(b, other_player)) {
            for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
              const typename Board::Pattern position{1LLU << i};
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

    std::vector<typename Board::Move> output_moves;
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

  std::vector<typename Board::Move> get_pruned_moves(const Board& b,
                                                     Player evalPlayer) {
    std::vector<typename Board::Move> candidates = get_moves(b, evalPlayer);
    std::size_t i = 1;
    while (i < candidates.size() &&
           abs(candidates[0].val - candidates[i].val) < pruning_thresh) {
      ++i;
    }
    if (i < candidates.size())
      candidates.erase(candidates.begin() + i, candidates.end());
    return candidates;
  }

  typename Board::Move get_random_move(const Board& b) {
    std::vector<std::size_t> options;
    for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
      const typename Board::Pattern position{1LLU << i};
      if (b.contains_spaces(position)) {
        options.push_back(i);
      }
    }

    if (options.size() > 0) {
      return typename Board::Move(options[std::uniform_int_distribution<int>(
                                      0, options.size() - 1U)(engine)],
                                  0.0, b.active_player());
    } else {
      return typename Board::Move(0, 0.0, b.active_player());
    }
  }

  typename Board::Move get_best_move_bfs(std::shared_ptr<Node<Board>> game_tree,
                                         Player player) {
    if (noise_enabled && lapse(engine))
      return get_random_move(game_tree->get_board());

    std::vector<FeatureP> null_features;
    FeatureRemover f(noise_enabled ? feature_packs : null_features, engine);
    std::shared_ptr<Node<Board>> n = game_tree->select();
    typename Board::Move best_move;
    const std::size_t max_iterations = std::size_t(1.0 / gamma) + 1;
    {
      std::size_t t = 0;
      std::size_t iterations = 0;
      typename Board::Move old_best_move;
      while (iterations++ < max_iterations && t < stopping_thresh &&
             !game_tree->determined()) {
        const std::vector<typename Board::Move> candidate_moves =
            get_pruned_moves(n->get_board(), player);
        n->expand(candidate_moves);
        n = game_tree->select();
        old_best_move = best_move;
        best_move = game_tree->get_best_move();
        old_best_move.board_position == best_move.board_position ? ++t : t = 0;
      }
    }

    return best_move;
  }

  typename Board::Move get_best_move_bfs(const Board& b, Player player) {
    return get_best_move_bfs(Node<Board>::create(b, evaluate(b)), player);
  }

  void set_noise_enabled(bool enabled) { noise_enabled = enabled; }

 private:
  /**
   * RAII class to remove and restore random features from a given heuristic.
   */
  class FeatureRemover {
   public:
    /**
     * Constructor. Removes random features.
     *
     * @param h The heuristic to remove random features from.
     */
    FeatureRemover(std::vector<FeatureP>& features, std::mt19937_64& engine)
        : features(features) {
      for (std::size_t i = 0; i < features.size(); ++i) {
        removed_features.push_back({});
        for (auto it = features[i].features.begin();
             it != features[i].features.end();) {
          if (std::bernoulli_distribution{features[i].drop_rate}(engine)) {
            removed_features[i].push_back(*it);
            features[i].features.erase(it);
          } else {
            ++it;
          }
        }
      }
    }

    /**
     * Destructor. Restores the features previously removed.
     */
    ~FeatureRemover() {
      for (std::size_t i = 0; i < removed_features.size(); ++i) {
        for (const auto& feature : removed_features[i]) {
          features[i].features.push_back(feature);
        }
      }
    }

   private:
    std::vector<FeatureP>& features;
    std::vector<std::vector<Feature>> removed_features;
  };
};

}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_H_INCLUDED
