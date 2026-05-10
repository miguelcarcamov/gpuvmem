/* Command-line options for gpuvmem (parse_gpuvmem_cli, print_help, goToError). */
#include "cli/gpuvmem_cli_config.hh"
#include "classes/flags.cuh"
#include "framework/vars.hh"

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
  v.blockSizeV = -1;
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

/* Short/long flags match historical gpuvmem CLI (tests + README). */
static void addOptions(Flags& f, Vars& v, GpuvmemCliRuntimeFlags& rf) {
  f.Var(v.input, 'i', "input", std::string("NULL"), "Input MS path(s)", "Required");
  f.Var(v.output, 'o', "output", std::string("NULL"), "Output MS path(s)", "Required");
  f.Var(v.modin, 'm', "model_input", std::string("NULL"), "Model FITS input", "Required");
  f.Var(v.path, 'p', "path", std::string("NULL"), "Output directory for FITS", "Optional");
  f.Var(v.output_image, 'O', "output_image", std::string("NULL"), "Output image basename", "Optional");
  f.Var(v.gpus, 'G', "gpus", std::string("NULL"), "Comma-separated GPU indices", "Optional");
  f.Var(v.initial_values, 'z', "initial_values", std::string("NULL"), "Comma-separated initial values", "Required");
  f.Var(v.stokes, 'S', "stokes", std::string(""), "Stokes to image (e.g. I,Q,U,V)", "Optional");
  f.Var(v.penalization_factors, 'Z', "regularization_factors", std::string("NULL"),
        "Comma-separated regularization weights", "Optional");
  f.Var(v.user_mask, 'u', "user_mask", std::string("NULL"), "User mask path", "Optional");
  f.Var(v.blockSizeX, 'X', "blockSizeX", -1, "GPU block size X", "Optional");
  f.Var(v.blockSizeY, 'Y', "blockSizeY", -1, "GPU block size Y", "Optional");
  f.Var(v.blockSizeV, 'V', "blockSizeV", -1,
        "GPU block size V (1D / visibility-style kernels; -1 = auto)", "Optional");
  f.Var(v.it_max, 't', "iterations", 1000, "Maximum iterations", "Optional");
  f.Var(v.gridding, 'g', "gridding", 1, "Gridding / CPU threads", "Optional");
  f.Var(v.noise, 'n', "noise", 0.0f, "Noise factor", "Optional");
  f.Var(v.noise_cut, 'N', "noise_cut", 0.0f, "Noise cut", "Optional");
  f.Var(v.randoms, 'r', "random_sampling", 1.0f, "Random sampling fraction", "Optional");
  f.Var(v.eta, 'e', "eta", -1.0f, "Entropy / positivity eta", "Optional");
  f.Var(v.nu_0, 'F', "ref_frequency", 0.0f, "Reference frequency (Hz)", "Optional");
  f.Var(v.robust_param, 'R', "robust_parameter", 0.0f, "Robust weighting parameter", "Optional");
  f.Var(v.threshold, 'T', "threshold", 0.0f, "Threshold (spectral index)", "Optional");
  f.Var(v.alpha_n_sigma, 'A', "alpha_n_sigma", 5.0f, "Alpha N-sigma mask", "Optional");
  f.Var(v.optimization_mode, 'J', "optimization_mode", std::string("joint"),
        "joint|block|one|alpha_static", "Optional");

  f.Bool(v.normalize, '\0', "normalize", "Normalize chi2 / precompute Neff", "Optional");

  f.Bool(rf.verbose, 'v', "verbose", "Verbose logging", "Flags");
  f.Bool(rf.nopositivity, 'x', "nopositivity", "Disable positivity", "Flags");
  f.Bool(rf.apply_noise, 'a', "apply-noise", "Apply Gaussian noise to visibilities", "Flags");
  f.Bool(rf.print_images, 'P', "print-images", "Print FITS each iteration", "Flags");
  f.Bool(rf.print_errors, 'E', "print-errors", "Print error maps", "Flags");
  f.Bool(rf.save_model_input, 's', "save_modelcolumn", "Write model to MODEL column", "Flags");
  f.Bool(rf.radius_mask, 'M', "use-radius-mask", "Radius mask instead of noise mask", "Flags");
  f.Bool(rf.modify_weights, '\0', "modify-weights", "Modify weights", "Flags");
}

static const GpuvmemCliRuntimeFlags* g_gpuvmem_cli_runtime = nullptr;

const GpuvmemCliRuntimeFlags* gpuvmem_cli_runtime_ptr() {
  return g_gpuvmem_cli_runtime;
}

void gpuvmem_cli_runtime_bind(const GpuvmemCliRuntimeFlags* runtime) {
  g_gpuvmem_cli_runtime = runtime;
}

bool parse_gpuvmem_cli(int argc, char** argv, GpuvmemCliConfig& out, std::ostream& err) {
  setDefaultVars(out.vars);
  out.runtime = GpuvmemCliRuntimeFlags{};
  out.argv0 = (argc > 0 && argv != nullptr && argv[0] != nullptr) ? std::string(argv[0]) : std::string{};
  Flags f;
  addOptions(f, out.vars, out.runtime);
  if (!f.Parse(argc, argv)) {
    err << "Invalid command line.\n";
    return false;
  }
  return true;
}

void print_help() {
  Vars v;
  setDefaultVars(v);
  GpuvmemCliRuntimeFlags rf{};
  Flags f;
  addOptions(f, v, rf);
  f.PrintHelp(std::cout);
  std::exit(1);
}

void goToError() {
  std::exit(1);
}
