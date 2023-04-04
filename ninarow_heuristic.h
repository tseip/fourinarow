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

/**
 * A heuristic for games of n-in-a-row.
 */
template <typename Board>
class Heuristic {
 public:
  using Feature = HeuristicFeature<Board>;

 private:
  static constexpr std::size_t NUM_PARAMS = 3 * Feature::Nweights + 7;

  double stopping_thresh;
  double pruning_thresh;
  double gamma;
  double lapse_rate;
  double opp_scale;
  double exploration_constant;
  double c_self, c_opp;
  double center_weight;
  std::array<double, Feature::Nweights> w_act;
  std::array<double, Feature::Nweights> w_pass;
  std::array<double, Feature::Nweights> delta;
  std::mt19937_64 engine;
  std::vector<Feature> features;
  std::array<double, Board::get_board_size()> vtile;
  std::normal_distribution<double> noise;
  std::bernoulli_distribution lapse;
  bool noise_enabled;

 public:
  Heuristic(const std::vector<Feature>& features)
      : stopping_thresh(7),
        pruning_thresh(5),
        gamma(0.01),
        lapse_rate(0.01),
        opp_scale(1),
        exploration_constant(0.0),
        c_self(0.0),
        c_opp(0.0),
        center_weight(1.0),
        w_act{0.8, 0.2, 3.5, 6,   0.8, 0.2, 3.5, 6, 0.8,
              0.2, 3.5, 6,   0.8, 0.2, 3.5, 6,   0},
        w_pass{0.8, 0.2, 3.5, 6,   0.8, 0.2, 3.5, 6, 0.8,
               0.2, 3.5, 6,   0.8, 0.2, 3.5, 6,   0},
        delta{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
        engine(),
        features(features),
        vtile(),
        noise(),
        lapse(),
        noise_enabled(true) {
    update();
  }

  Heuristic(const std::string filename, const std::vector<Feature>& features,
            std::size_t skiplines = 0)
      : Heuristic(features) {
    std::ifstream input;
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      input.open(filename, std::ios::in);

      // Skip header lines.
      std::string s;
      for (std::size_t i = 0; i < skiplines; ++i) std::getline(input, s);

      // Read in parameters.
      std::vector<double> params;
      for (std::size_t i = 0; i < NUM_PARAMS; ++i) {
        double param = 0.0;
        input >> param;
        params.push_back(param);
      }
      get_params_from_vector(params);
    } catch (const std::ios_base::failure& /*fail*/) {
      input.close();
      throw;
    }
    input.close();
  }

  Heuristic(const std::vector<double>& params,
            const std::vector<Feature>& features)
      : Heuristic(features) {
    get_params_from_vector(params);
  }

  Heuristic(double feature_drop_slider, double lapse_slider,
            double value_noise_slider, double search_slider,
            double offense_slider, const std::vector<Feature>& features)
      : Heuristic(features) {
    const double a = 2.0 * value_noise_slider;
    const double b = 2.0 - a;
    const double w_center = 0.6;
    const double w_2conn = 0.9;
    const double w_2unc = 0.45;
    const double w_3 = 3.5;
    const double w_4 = 6.0;
    stopping_thresh = 10000.0;
    pruning_thresh = a * 4.0;
    gamma = pow(0.001, search_slider);
    lapse_rate = 1.0 - lapse_slider;
    opp_scale = 0.25 * pow(16, offense_slider);
    exploration_constant = 1.0;
    center_weight = a * w_center;
    for (unsigned int i = 0; i < Feature::Nweights; i += 4) {
      w_act[i] = a * w_2conn;
      w_pass[i] = a * w_2conn;
      w_act[i + 1] = a * w_2unc;
      w_pass[i + 1] = a * w_2unc;
      w_act[i + 2] = a * w_3;
      w_pass[i + 2] = a * w_3;
      w_act[i + 3] = a * w_4;
      w_pass[i + 3] = a * w_4;
    }
    w_act[Feature::Nweights - 1] = 0.0;
    w_pass[Feature::Nweights - 1] = 0.0;
    for (unsigned int i = 0; i < Feature::Nweights; i++)
      delta[i] = 1.0 - feature_drop_slider;
    update();
    noise = std::normal_distribution<double>(0.0, b);
  }

 private:
  void get_params_from_vector(const std::vector<double>& params) {
    if (params.size() != NUM_PARAMS) {
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
    for (std::size_t j = 0; j < Feature::Nweights; ++j) w_act[j] = params[i++];
    for (std::size_t j = 0; j < Feature::Nweights; ++j) w_pass[j] = params[i++];
    for (std::size_t j = 0; j < Feature::Nweights; ++j) delta[j] = params[i++];
    update();
  }

  void update() {
    noise = std::normal_distribution<double>(0.0, 1.0);
    lapse = std::bernoulli_distribution(lapse_rate);
    for (std::size_t i = 0; i < Board::get_board_size(); ++i)
      vtile[i] = 1.0 / sqrt(pow(i / Board::get_board_width() - 1.5, 2) +
                            pow(i % Board::get_board_width() - 4.0, 2));
    c_self = 2.0 * opp_scale / (1.0 + opp_scale);
    c_opp = 2.0 / (1.0 + opp_scale);
    for (auto& feature : features) {
      feature.update_weights(w_act[feature.weight_index],
                             w_pass[feature.weight_index],
                             delta[feature.weight_index]);
    }
  }

 public:
  void seed_generator(uint64_t seed) { engine.seed(seed); }

  double evaluate(const Board& b) const {
    const Player player = b.active_player();
    double val = 0.0;
    for (std::size_t i = 0; i < Board::get_board_size(); ++i) {
      const typename Board::Pattern pattern{1LLU << i};
      if (b.contains(pattern, player)) val += center_weight * vtile[i];
      if (b.contains(pattern, get_other_player(player)))
        val -= center_weight * vtile[i];
    }
    for (const auto& feature : features) {
      if (feature.is_active(b, player) && feature.contained(b, player)) {
        val += feature.weight_act;
      } else if (feature.is_active(b, get_other_player(player)) &&
                 feature.contained(b, get_other_player(player))) {
        val -= feature.weight_pass;
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
    for (const auto& feature : features) {
      if (feature.contained(b, player) && feature.is_active(b, player))
        deltaL -= c_pass * feature.diff_act_pass();
      else if (feature.contained(b, other_player) &&
               feature.is_active(b, other_player))
        deltaL -= c_act * feature.diff_act_pass();
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

    for (const auto& feature : features) {
      if (feature.is_active(b, player) || feature.is_active(b, other_player)) {
        const typename Board::Pattern player_missing_pieces =
            feature.missing_pieces(b, player);
        const typename Board::Pattern other_player_missing_pieces =
            feature.missing_pieces(b, other_player);
        if ((player_missing_pieces.count_overlap(other_player_missing_pieces) !=
             0) &&
            player_missing_pieces.positions.count() == 1) {
          auto search = candidate_moves.find(player_missing_pieces);
          if (search != candidate_moves.end()) {
            search->second.val += c_pass * feature.weight_pass;
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
                search->second.val -= c_pass * feature.weight_pass;
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
                search->second.val += c_act * feature.weight_act;
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

    std::vector<Feature> null_features;
    FeatureRemover f(noise_enabled ? features : null_features, engine);
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
    FeatureRemover(std::vector<Feature>& features, std::mt19937_64& engine)
        : features(features) {
      for (auto it = features.begin(); it != features.end();) {
        if (std::bernoulli_distribution{it->drop_rate}(engine)) {
          removed_features.push_back(*it);
          features.erase(it);
        } else {
          ++it;
        }
      }
    }

    /**
     * Destructor. Restores the features previously removed.
     */
    ~FeatureRemover() {
      for (const auto& feature : removed_features) {
        features.push_back(feature);
      }
    }

   private:
    std::vector<Feature>& features;
    std::vector<Feature> removed_features;
  };
};

}  // namespace NInARow

#endif  // NINAROW_HEURISTIC_H_INCLUDED
