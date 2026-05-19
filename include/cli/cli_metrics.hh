#ifndef GPUVMEM_CLI_CLI_METRICS_HH
#define GPUVMEM_CLI_CLI_METRICS_HH

#include <string>
#include <vector>

namespace gpuvmem {
namespace cli {

/** Per-Fi line in run summary / final metrics (presentation-agnostic). */
struct ObjectiveTermLine {
  std::string name;
  double lambda = 0.0;
  bool active = false;
};

/** Astronomy + setup snapshot emitted once before optimization. */
struct RunSummary {
  int n_ms = 0;
  std::vector<std::string> input_ms;
  std::vector<std::string> output_ms;
  std::string model_description;
  std::string radesys;
  double equinox = 0.0;
  double phase_ra_deg = 0.0;
  double phase_dec_deg = 0.0;
  long grid_m = 0;
  long grid_n = 0;
  double cell_arcsec = 0.0;
  double ref_col = 0.0;
  double ref_row = 0.0;
  std::string planes_description;
  double freq_min_hz = 0.0;
  double freq_max_hz = 0.0;
  double nu0_hz = 0.0;
  double log_nu_min = 0.0;
  double log_nu_max = 0.0;
  int vis_rows_used = 0;
  double random_sample_frac = 1.0;
  bool gridding = false;
  std::string weighting;
  double robust_r = 0.0;
  double max_uv_wavelength = 0.0;
  double nyquist_cell_arcsec = 0.0;
  double beam_major_arcsec = 0.0;
  double beam_minor_arcsec = 0.0;
  double beam_pa_deg = 0.0;
  double noise_jy_per_pixel = 0.0;
  double fg_scale = 0.0;
  std::vector<ObjectiveTermLine> objective_terms;
  std::string optimizer_id;
  std::string optimization_mode;
  int max_iter = 0;
  double ftol = 0.0;
  double gtol = 0.0;
  std::string linesearch_id;
  std::string seeder_id;
  int lbfgs_m = 0;
  bool normalize_chi2 = false;
  double minpix = 0.0;
  double noise_cut_eff = 0.0;
  std::string mask_description;
  int gpu_first = 0;
  int gpu_count = 0;
  std::string gpu_name;
  double gpu_mem_gib = 0.0;
};

/** One optimizer iteration (after line search at accepted point). */
struct IterationMetrics {
  int iteration = 0;
  int max_iterations = 0;
  float phi = 0.f;
  float chi2_w = 0.f;
  float reg_w = 0.f;
  double wall_time_s = 0.0;
  int sub_run = 0;
  int sub_run_total = 0;
  std::string sub_plane_label;
};

struct OptimizationBeginInfo {
  std::string optimizer_id;
  std::string mode_description;
  int max_iterations = 0;
};

enum class OptimizationStopReason {
  MaxIterations,
  FunctionTolerance,
  GradientTolerance,
  ObjectivePlateau,
};

struct OptimizationEndInfo {
  OptimizationStopReason reason = OptimizationStopReason::MaxIterations;
  int iteration = 0;
  std::string detail;
};

struct FiMetricLine {
  std::string name;
  double lambda = 0.0;
  double value = 0.0;
  double lambda_times_value = 0.0;
};

/** Post-optimization numbers (stdout / --metrics-file). */
struct FinalRunMetrics {
  float phi_total = 0.f;
  float chi2_w = 0.f;
  float reg_w = 0.f;
  std::string optimizer_id;
  std::string optimization_mode;
  double ftol = 0.0;
  double gtol = 0.0;
  int iteration_budget = 0;
  int iteration_last = 0;
  std::vector<FiMetricLine> fi_terms;
  double cpu_time_s = 0.0;
  double wall_time_s = 0.0;
};

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_CLI_METRICS_HH
