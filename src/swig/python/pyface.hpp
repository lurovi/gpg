#include "../../util.hpp"
#include "../../myeig.hpp"
#include "../../globals.hpp"
#include "../../evolution.hpp"


using namespace std;

Node * best = NULL;
Evolution * evo = NULL;

pair<myeig::Mat, myeig::Vec> _assemble_Xy(double * X_n_y, int n_obs, int n_feats_plus_label) {
  myeig::Mat X(n_obs, n_feats_plus_label);
  myeig::Vec y(n_obs);
  for(int i = 0; i < n_obs; i++) {
    double * row = &X_n_y[i*(n_feats_plus_label)];
    for(int j = 0; j < n_feats_plus_label - 1; j++) {
      X(i,j) = (float) row[j];
    }
    y(i) = X_n_y[i*(n_feats_plus_label)+n_feats_plus_label-1];
  }
  return make_pair(X,y);
}

void _include_prediction_back(Vec & prediction, double * X_n_p, int n_feats_plus_one) {
  int n_obs = prediction.size();
  
  for(int i = 0; i < n_obs; i++) {
    X_n_p[i*(n_feats_plus_one)+n_feats_plus_one-1] = (double) prediction(i);
  }
}

void setup(char * options) {
  string str_options = string(options);
  auto opts = split_string(str_options, " ");
  int argc = opts.size()+1;
  char * argv[argc];
  argv[0] = "minigpg";
  for (int i = 1; i < argc; i++) {
    argv[i] = (char*) opts[i-1].c_str();
  }
  g::read_options(argc, argv);
  evo = new Evolution();
}

void fit(double * X_n_y, int n_obs, int n_feats_plus_label) {
  auto Xy = _assemble_Xy(X_n_y, n_obs, n_feats_plus_label);
  auto X = Xy.first;
  auto y = Xy.second;

  // continue
  g::fit_func->set_Xy(X, y);
  evo->run();
  print(g::max_generations);
}

void predict(double * X_n_p, int n_obs, int n_feats_plus_one) {
  if (!evo->elite) {
    print("Not fitted");
    //throw runtime_error("Not fitted");
  }
  // assemble Mat
  myeig::Mat X = _assemble_Xy(X_n_p, n_obs, n_feats_plus_one).first;

  // mock prediction
  myeig::Vec prediction = evo->elite->get_output(X);
  _include_prediction_back(prediction, X_n_p, n_feats_plus_one);

  return;
}