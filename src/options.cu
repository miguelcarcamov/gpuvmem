/* Command-line options for gpuvmem (parse_gpuvmem_cli, print_help, goToError). */
#include "cli/gpuvmem_cli_config.hh"
#include "classes/flags.cuh"
#include "framework/vars.hh"

#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

static void setDefaultVars(Vars& v) {
  v.input = "NULL";
  v.output = "NULL";
  v.modin = "NULL";
  v.ofile = "NULL";
  v.path = "NULL";
  v.output_image = "NULL";
  v.gpus = "NULL";
  v.initial_values = "NULL";
  v.initial_model_fits = "NULL";
  v.stokes = "";
  v.regularization_weights = "NULL";
  v.user_mask = "NULL";
  v.blockSizeX = -1;
  v.blockSizeY = -1;
  v.blockSizeV = -1;
  v.it_max = 1000;
  v.gridding = 1;
  v.noise = 0.0f;
  /* Legacy default (chg/ai-changes src/functions.cu Var 'N'): 10.0 — scaled by
   * noise map minimum in MFS; 0 would make noise>=cut true for all non-negative
   * noise/distance pixels and zero the χ² gradient. */
  v.noise_cut = 10.0f;
  v.randoms = 1.0f;
  v.eta = -1.0f;
  v.nu_0 = 0.0f;
  v.robust_param = 0.0f;
  v.threshold = 0.0f;
  v.alpha_n_sigma = 5.0f;
  v.normalize = false;
  v.optimization_mode = "joint";
  v.weighting_scheme = "Natural";
  v.optimizer_name = "LBFGS";
  v.linesearch_name.clear();
  v.seeder_name.clear();
  v.lbfgs_corrections = 10;
  v.imsize.clear();
  v.cellsize_arcsec = 0.0f;
  v.phase_center_deg.clear();
}

static std::string trimToken(const std::string& s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

static bool parseCommaPair(const std::string& spec, double& a, double& b, std::ostream& err,
                           const char* label) {
  const size_t comma = spec.find(',');
  if (comma == std::string::npos) {
    err << label << " must be two comma-separated numbers (got: \"" << spec << "\").\n";
    return false;
  }
  std::string sa = trimToken(spec.substr(0, comma));
  std::string sb = trimToken(spec.substr(comma + 1));
  if (sa.empty() || sb.empty()) {
    err << label << " has empty component (got: \"" << spec << "\").\n";
    return false;
  }
  std::istringstream isa(sa), isb(sb);
  isa >> a;
  isb >> b;
  if (isa.fail() || isb.fail()) {
    err << label << " could not parse numbers (got: \"" << spec << "\").\n";
    return false;
  }
  return true;
}

static bool validate_gpuvmem_cli_vars(const Vars& v, std::ostream& err) {
  if (!v.imsize.empty()) {
    double dm = 0.0, dn = 0.0;
    if (!parseCommaPair(v.imsize, dm, dn, err, "--imsize M,N")) return false;
    if (dm != std::floor(dm) || dn != std::floor(dn) || dm <= 0.0 || dn <= 0.0) {
      err << "--imsize M,N requires positive integer M and N.\n";
      return false;
    }
  }
  if (!v.phase_center_deg.empty()) {
    double ra = 0.0, dec = 0.0;
    if (!parseCommaPair(v.phase_center_deg, ra, dec, err, "--phase-center RA_DEG,DEC_DEG"))
      return false;
    (void)ra;
    (void)dec;
  }
  const bool no_model =
      (v.modin == "NULL" || v.modin.empty() || v.modin == "NONE" || v.modin == "-");
  const bool synth_ok =
      !v.imsize.empty() && v.cellsize_arcsec > 0.0f && !v.phase_center_deg.empty();
  if (no_model && !synth_ok) {
    err << "Provide model FITS (-m), or synthetic grid: --imsize M,N --cellsize ARCSEC "
           "--phase-center RA_DEG,DEC_DEG.\n";
    return false;
  }
  if (!no_model && synth_ok) {
    // FITS wins; synthetic flags ignored silently (avoid surprising overrides).
  }
  if (v.initial_model_fits != "NULL" && !v.initial_model_fits.empty()) {
    struct stat st {};
    if (stat(v.initial_model_fits.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      err << "--initial-model must be a readable regular file (got: \"" << v.initial_model_fits
          << "\").\n";
      return false;
    }
  }
  return true;
}

static bool parse_progress_mode(const std::string& s, GpuvmemCliRuntimeFlags::ProgressModeKind& out,
                                std::ostream& err) {
  if (s == "auto") {
    out = GpuvmemCliRuntimeFlags::ProgressAuto;
    return true;
  }
  if (s == "bar") {
    out = GpuvmemCliRuntimeFlags::ProgressBar;
    return true;
  }
  if (s == "plain") {
    out = GpuvmemCliRuntimeFlags::ProgressPlain;
    return true;
  }
  if (s == "off") {
    out = GpuvmemCliRuntimeFlags::ProgressOff;
    return true;
  }
  err << "--progress must be auto, bar, plain, or off (got: \"" << s << "\").\n";
  return false;
}

/* Group names use a numeric prefix so --help sections follow a sensible order (std::map is sorted). */
static void addOptions(Flags& f, Vars& v, GpuvmemCliRuntimeFlags& rf) {
  f.Var(v.input, 'i', "input", std::string("NULL"),
        "Comma-separated input Measurement Set path(s).", "1 Core (required)");
  f.Var(v.output, 'o', "output", std::string("NULL"),
        "Comma-separated output MS path(s); one per input.", "1 Core (required)");
  f.Var(v.initial_values, 'z', "initial_values", std::string("NULL"),
        "Comma-separated per-image constants; first is MINPIX (positivity floor). "
        "Optional --initial-model replaces plane 0 from FITS; other planes use -z.",
        "1 Core (required)");
  f.Var(v.modin, 'm', "model_input", std::string("NULL"),
        "Model FITS: image size, CDELT, CRVAL, CRPIX-derived ref. pixel. Omit if using "
        "--imsize, --cellsize, --phase-center instead.", "2 Model / WCS");
  f.Var(v.imsize, '\0', "imsize", std::string(""),
        "Synthetic grid only: M,N as NAXIS2,NAXIS1 (rows,cols). Requires --cellsize and "
        "--phase-center.", "2 Model / WCS");
  f.Var(v.cellsize_arcsec, '\0', "cellsize", 0.0f,
        "Synthetic grid: square pixel side in arcseconds (with --imsize).", "2 Model / WCS");
  f.Var(v.phase_center_deg, '\0', "phase-center", std::string(""),
        "Synthetic grid: RA,DEC in degrees ICRS (comma-separated).", "2 Model / WCS");
  f.Var(v.initial_model_fits, 'I', "initial-model", std::string("NULL"),
        "FITS float 2D HDU: overwrite plane-0 pixels after -z fill. Grid must match model. "
        "-I avoids clash with -i.", "2 Model / WCS");
  f.Var(v.stokes, 'S', "stokes", std::string(""),
        "Comma-separated Stokes to reconstruct (e.g. I or I,Q,U,V).", "2 Model / WCS");

  f.Var(v.weighting_scheme, 'W', "weighting", std::string("Natural"),
        "Factory id: Natural|Uniform|Radial|Briggs. CLI may use lower case; robust maps to Briggs. "
        "-R is the Briggs robustness parameter.", "3 Weighting & data");
  f.Var(v.robust_param, 'R', "robust_parameter", 0.0f, "Briggs robust parameter (with -W Briggs or robust).",
        "3 Weighting & data");
  f.Var(v.noise, 'n', "noise", 0.0f, "Visibility noise scale (see code / docs).", "3 Weighting & data");
  f.Var(v.noise_cut, 'N', "noise_cut", 10.0f,
        "Chi2 mask cut; scaled by min noise map value in MFS (legacy default 10).", "3 Weighting & data");
  f.Var(v.randoms, 'r', "random_sampling", 1.0f, "Fraction of rows to read (0-1].", "3 Weighting & data");
  f.Var(v.gridding, 'g', "gridding", 1, "Gridding on (1) or use MS as-is (0); also thread hint.",
        "3 Weighting & data");
  f.Var(v.nu_0, 'F', "ref_frequency", 0.0f,
        "Reference frequency nu_0 (Hz); 0 lets the code pick band centre.", "3 Weighting & data");
  f.Var(v.user_mask, 'u', "user_mask", std::string("NULL"), "External FITS mask path.", "3 Weighting & data");
  f.Bool(v.normalize, '\0', "normalize", "Normalize chi2 and precompute N_eff.", "3 Weighting & data");

  f.Var(v.optimizer_name, 'C', "optimizer", std::string("LBFGS"),
        "Factory id (case-sensitive): LBFGS; CG-PolakRibiere, CG-HagerZhang, CG-DaiYuan, "
        "CG-FletcherReeves, CG-HestenesStiefel, CG-LiuStorey, CG-RMIL.",
        "4 Optimization");
  f.Var(v.linesearch_name, 'L', "linesearch", std::string(""),
        "Line search factory: Brent, GoldenSectionSearch, FibonacciSearch, GLLArmijo, "
        "BacktrackingArmijo, FistaBacktracking, Fixed. Default when omitted: LBFGS/CG use "
        "built-in Brent (see optimizers' constructors; main does not replace -L if empty).",
        "4 Optimization");
  f.Var(v.seeder_name, 'B', "seeder", std::string(""),
        "Step seeder: BBMin1Seeder, BBMin2Seeder, BBAlternatingSeeder, QuadraticInterpolationSeeder, "
        "CubicInterpolationSeeder. Default when omitted: none. If you pass -B without -L, "
        "main selects Brent for -L so the seeder is attached.",
        "4 Optimization");
  f.Var(v.lbfgs_corrections, 'K', "lbfgs-m", 10, "L-BFGS history size M (LBFGS only).", "4 Optimization");
  f.Var(v.optimization_mode, 'J', "optimization_mode", std::string("joint"),
        "joint | block | one | alpha_static (multi-plane / MFS modes).", "4 Optimization");
  f.Var(v.it_max, 't', "iterations", 1000, "Maximum optimizer iterations.", "4 Optimization");
  f.Var(v.eta, 'e', "eta", -1.0f, "Entropy / positivity eta (must be < 0 for clipping; default -1).",
        "4 Optimization");
  f.Var(v.threshold, 'T', "threshold", 0.0f, "Spectral-index related threshold (scaled internally).",
        "4 Optimization");
  f.Var(v.alpha_n_sigma, 'A', "alpha_n_sigma", 5.0f, "Spectral index mask width in sigma.", "4 Optimization");
  f.Var(v.regularization_weights, 'Z', "regularization_factors", std::string("NULL"),
        "Comma-separated weights: <5 values → fixed Chi2 λ=1, then Entropy, L1, TSV. "
        "≥5 values → first is Chi2, then Entropy, L1, TSV (extra entries ignored).",
        "4 Optimization");

  f.Var(v.path, 'p', "path", std::string("NULL"), "Directory for written FITS products.", "5 Output & metrics");
  f.Var(v.output_image, 'O', "output_image", std::string("NULL"), "Basename for main output FITS.", "5 Output & metrics");
  f.Var(v.ofile, '\0', "metrics-file", std::string("NULL"),
        "Write run summary (phi, per-Fi lambda*value, times) to this path.", "5 Output & metrics");

  f.Var(v.gpus, 'G', "gpus", std::string("NULL"), "Comma-separated CUDA device indices.", "6 GPU tuning");
  f.Var(v.blockSizeX, 'X', "blockSizeX", -1, "CUDA block dim X (-1 = auto).", "6 GPU tuning");
  f.Var(v.blockSizeY, 'Y', "blockSizeY", -1, "CUDA block dim Y (-1 = auto).", "6 GPU tuning");
  f.Var(v.blockSizeV, 'V', "blockSizeV", -1, "CUDA block for 1D kernels (-1 = auto).", "6 GPU tuning");

  f.Bool(rf.verbose, 'v', "verbose", "Extra MS/WCS detail and optimizer diagnostics.", "7 Flags");
  f.Bool(rf.quiet, 'q', "quiet", "Minimal stdout (errors + short final line).", "7 Flags");
  f.Bool(rf.debug, '\0', "debug", "GPU/CUDA grid detail and internal optimizer messages.", "7 Flags");
  f.Var(rf.progress_mode_user, '\0', "progress", std::string("auto"),
        "Optimization progress: auto|bar|plain|off (auto uses a bar on TTY).", "7 Flags");
  f.Var(rf.log_interval, '\0', "log-interval", 1,
        "Print every N optimizer iterations in plain/auto file mode.", "7 Flags");
  f.Bool(rf.nopositivity, 'x', "nopositivity", "Turn off positivity projection in line search.", "7 Flags");
  f.Bool(rf.apply_noise, 'a', "apply-noise", "Add Gaussian noise to simulated / stored visibilities.", "7 Flags");
  f.Bool(rf.print_images, 'P', "print-images", "Write intermediate FITS each iteration.", "7 Flags");
  f.Bool(rf.print_errors, 'E', "print-errors", "Write uncertainty / error maps where configured.", "7 Flags");
  f.Bool(rf.save_model_input, 's', "save_modelcolumn", "Populate MODEL_DATA in output MS.", "7 Flags");
  f.Bool(rf.radius_mask, 'M', "use-radius-mask", "Use radial distance mask instead of noise map.", "7 Flags");
  f.Bool(rf.modify_weights, '\0', "modify-weights", "Apply weighting-scheme modifications to weights.",
        "7 Flags");

  f.Bool(rf.print_warranty, 'w', "warranty", "Print GPL warranty text and exit (no MS run).", "8 Informational");
  f.Bool(rf.print_copyright, 'c', "copyright", "Print GPL copying terms and exit (no MS run).",
        "8 Informational");
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
  // Legacy src/functions.cu: user-supplied FITS mask forces noise_cut=1.0
  if (out.vars.user_mask != "NULL" && !out.vars.user_mask.empty()) {
    out.vars.noise_cut = 1.0f;
  }
  // If user is clearly starting an imaging run (-i set), ignore informational -w/-c
  // so a stray -w does not skip validation or exit before CUDA.
  const bool have_input =
      !out.vars.input.empty() && out.vars.input != "NULL";
  if (have_input) {
    out.runtime.print_warranty = false;
    out.runtime.print_copyright = false;
  }
  if (out.runtime.print_warranty || out.runtime.print_copyright) return true;
  if (!validate_gpuvmem_cli_vars(out.vars, err)) return false;
  if (!parse_progress_mode(out.runtime.progress_mode_user, out.runtime.progress_mode, err))
    return false;
  if (out.runtime.log_interval < 1) {
    err << "--log-interval must be >= 1.\n";
    return false;
  }
  if (out.runtime.quiet && out.runtime.verbose) {
    err << "--quiet and --verbose are mutually exclusive.\n";
    return false;
  }
  return true;
}

void print_help(const char* program_name) {
  Vars v;
  setDefaultVars(v);
  GpuvmemCliRuntimeFlags rf{};
  Flags f;
  addOptions(f, v, rf);
  f.setProgramNameForHelp((program_name && program_name[0]) ? std::string(program_name)
                                                               : std::string("gpuvmem"));
  std::cout
      << "\nTypical use: -i input.ms[,...] -o output.ms[,...] -m model.fits -z MINPIX[,...]\n"
      << "Without -m: add --imsize M,N --cellsize ARCSEC --phase-center RA_DEG,DEC_DEG\n"
      << "Factory ids for -C, -L, -B, -W are case-sensitive (see numbered sections below).\n\n";
  f.PrintHelp(std::cout);
  std::exit(1);
}

void goToError() {
  std::exit(1);
}
