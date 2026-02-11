/* Command-line options for gpuvmem (getOptions, print_help, goToError). */
#include "main.cuh"
#include "framework.cuh"
#include <cstdlib>
#include <iostream>

static void setDefaultVars(Vars& v) {
  v.input = "NULL";
  v.output = "NULL";
  v.inputdat = "NULL";
  v.modin = "NULL";
  v.ofile = "NULL";
  v.path = "NULL";
  v.output_image = "NULL";
  v.gpus = "NULL";
  v.initial_values = "NULL";
  v.stokes = "";
  v.penalization_factors = "NULL";
  v.user_mask = "NULL";
  v.blockSizeX = -1;
  v.blockSizeY = -1;
  v.blockSizeV = 256;
  v.it_max = 1000;
  v.gridding = 1;
  v.noise = 0.0f;
  v.noise_cut = 0.0f;
  v.randoms = 1.0f;
  v.eta = -1.0f;
  v.nu_0 = 0.0f;
  v.robust_param = 0.0f;
  v.threshold = 0.0f;
  v.alpha_n_sigma = 5.0f;
  v.normalize = false;
  v.optimization_mode = "joint";
}

static void addOptions(Flags& f, Vars& v) {
  f.Var(v.input, 'i', "input", std::string("NULL"), "Input MS path(s)", "Required");
  f.Var(v.output, 'o', "output", std::string("NULL"), "Output MS path(s)", "Required");
  f.Var(v.modin, 'm', "modin", std::string("NULL"), "Model input", "Optional");
  f.Var(v.path, 'p', "path", std::string("NULL"), "Output path", "Optional");
  f.Var(v.output_image, 'O', "output_image", std::string("NULL"), "Output image name", "Optional");
  f.Var(v.gpus, 'g', "gpus", std::string("NULL"), "GPU device IDs", "Optional");
  f.Var(v.initial_values, 'z', "initial_values", std::string("NULL"), "Initial values (first=MINPIX)", "Required");
  f.Var(v.stokes, 's', "stokes", std::string(""), "Stokes to image", "Optional");
  f.Var(v.penalization_factors, 'l', "penalization_factors", std::string("NULL"), "Penalization factors", "Optional");
  f.Var(v.user_mask, 'u', "user_mask", std::string("NULL"), "User mask", "Optional");
  f.Var(v.blockSizeX, 'x', "blockSizeX", -1, "Block size X", "Optional");
  f.Var(v.blockSizeY, 'y', "blockSizeY", -1, "Block size Y", "Optional");
  f.Var(v.blockSizeV, 'v', "blockSizeV", 256, "Block size V", "Optional");
  f.Var(v.it_max, 'I', "it_max", 1000, "Max iterations", "Optional");
  f.Var(v.gridding, 'G', "gridding", 1, "Gridding threads", "Optional");
  f.Var(v.noise, 'n', "noise", 0.0f, "Noise level", "Optional");
  f.Var(v.noise_cut, 'c', "noise_cut", 0.0f, "Noise cut", "Optional");
  f.Var(v.randoms, 'r', "randoms", 1.0f, "Random probability", "Optional");
  f.Var(v.eta, 'e', "eta", -1.0f, "Eta", "Optional");
  f.Var(v.nu_0, 'f', "nu_0", 0.0f, "Reference frequency", "Optional");
  f.Var(v.robust_param, 'R', "robust_param", 0.0f, "Robust param", "Optional");
  f.Var(v.threshold, 't', "threshold", 0.0f, "Threshold", "Optional");
  f.Var(v.alpha_n_sigma, 'a', "alpha_n_sigma", 5.0f, "Alpha N sigma", "Optional");
  f.Var(v.optimization_mode, 'M', "optimization_mode", std::string("joint"), "Mode", "Optional");
  f.Bool(v.normalize, 'N', "normalize", "Normalize", "Optional");
}

Vars getOptions(int argc, char** argv) {
  Vars v;
  setDefaultVars(v);
  Flags f;
  addOptions(f, v);
  if (!f.Parse(argc, argv)) {
    print_help();
    std::exit(1);
  }
  return v;
}

void print_help() {
  Vars v;
  setDefaultVars(v);
  Flags f;
  addOptions(f, v);
  f.PrintHelp(std::cout);
  std::exit(1);
}

void goToError() {
  std::exit(1);
}
