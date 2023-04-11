// fourbynine.i - SWIG interface
%module fourbynine
%{
#include "ninarow_board.h"
#include "ninarow_move.h"
#include "ninarow_pattern.h"
#include "ninarow_heuristic.h"
#include "ninarow_heuristic_feature.h"
#include "fourbynine_features.h"
#include "player.h"
%}

%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_shared_ptr.i"

// Parse the original header files
%include "ninarow_board.h"
%include "ninarow_move.h"
%include "ninarow_pattern.h"
%include "ninarow_heuristic.h"
%include "ninarow_heuristic_feature.h"
%include "fourbynine_features.h"
%include "player.h"

// Instantiate some templates

%template(fourbynine_board) NInARow::Board<4, 9, 4>;
%template(fourbynine_move) NInARow::Move<4, 9, 4>;
%template(fourbynine_pattern) NInARow::Pattern<4, 9, 4>;
%template(fourbynine_heuristic_feature_pack) NInARow::FeaturePack<NInARow::Board<4, 9, 4>>;
%template(fourbynine_heuristic) NInARow::Heuristic<NInARow::Board<4, 9, 4>>;
%template(fourbynine_heuristic_feature) NInARow::HeuristicFeature<NInARow::Board<4, 9, 4>>;
%template(DoubleVector) std::vector<double>;
%template(FeaturePackVector) std::vector<NInARow::FeaturePack<NInARow::Board<4, 9, 4>>>;
%template(FeatureVector) std::vector<NInARow::HeuristicFeature<NInARow::Board<4, 9, 4>>>;
