#include "cli/console_run_observer.hh"

#include "cli/optimization_reporting.hh"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace gpuvmem {
namespace cli {

namespace {

std::string format_sci(double v) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(6) << v;
  return os.str();
}

std::string format_eta_seconds(double eta_s) {
  if (!std::isfinite(eta_s) || eta_s < 0.0) return "?";
  const int sec = static_cast<int>(eta_s + 0.5);
  if (sec < 60) return std::to_string(sec) + "s";
  if (sec < 3600) return std::to_string(sec / 60) + "m";
  return std::to_string(sec / 3600) + "h" + std::to_string((sec % 3600) / 60) + "m";
}

}  // namespace

LogLevel log_level_from_runtime(const GpuvmemCliRuntimeFlags& runtime) {
  if (runtime.quiet) return LogLevel::Quiet;
  if (runtime.debug) return LogLevel::Debug;
  if (runtime.verbose) return LogLevel::Verbose;
  return LogLevel::Normal;
}

ProgressMode progress_mode_from_runtime(const GpuvmemCliRuntimeFlags& runtime, bool is_tty) {
  switch (runtime.progress_mode) {
    case GpuvmemCliRuntimeFlags::ProgressBar:
      return ProgressMode::Bar;
    case GpuvmemCliRuntimeFlags::ProgressPlain:
      return ProgressMode::Plain;
    case GpuvmemCliRuntimeFlags::ProgressOff:
      return ProgressMode::Off;
    default:
      return is_tty ? ProgressMode::Bar : ProgressMode::Plain;
  }
}

ConsoleRunObserver::ConsoleRunObserver(const GpuvmemCliRuntimeFlags& runtime, std::ostream& out)
    : runtime_(runtime), out_(out) {}

bool ConsoleRunObserver::should_print_iteration(int iteration, int max_iterations) const {
  const int interval = std::max(1, runtime_.log_interval);
  return iteration % interval == 0 || iteration == max_iterations;
}

void ConsoleRunObserver::on_diagnostic(DiagnosticLevel level, const std::string& message) {
  const LogLevel ll = log_level_from_runtime(runtime_);
  if (level == DiagnosticLevel::Debug && ll < LogLevel::Debug) return;
  if (level == DiagnosticLevel::Verbose && ll < LogLevel::Verbose) return;
  out_ << message;
  if (!message.empty() && message.back() != '\n') out_ << '\n';
}

void ConsoleRunObserver::on_run_summary(const RunSummary& ctx) {
  if (log_level_from_runtime(runtime_) == LogLevel::Quiet) return;

  out_ << "\n=== gpuvmem run summary ===\n";
  out_ << "inputs: " << ctx.n_ms << " MS\n";
  for (int i = 0; i < ctx.n_ms; ++i) {
    const std::string& in = (i < static_cast<int>(ctx.input_ms.size())) ? ctx.input_ms[i] : "?";
    const std::string& o = (i < static_cast<int>(ctx.output_ms.size())) ? ctx.output_ms[i] : "?";
    out_ << "  [" << i << "] " << in << " -> " << o << '\n';
  }
  out_ << "model: " << ctx.model_description << '\n';
  out_ << std::fixed << std::setprecision(6);
  out_ << "sky: RADESYS=" << ctx.radesys << " EQUINOX=" << ctx.equinox << "  phase_center_RA_deg="
      << ctx.phase_ra_deg << "  phase_center_DEC_deg=" << ctx.phase_dec_deg << '\n';
  out_ << "grid: M=" << ctx.grid_m << " N=" << ctx.grid_n << "  cell_arcsec=" << ctx.cell_arcsec
      << "  ref_pix_0based_col,row=(" << ctx.ref_col << ',' << ctx.ref_row << ")\n";
  out_ << "planes: " << ctx.planes_description << '\n';
  out_ << std::scientific << std::setprecision(6);
  out_ << "data: band_Hz=[" << ctx.freq_min_hz << ',' << ctx.freq_max_hz << "]  nu0_Hz=" << ctx.nu0_hz
      << "  log_nu_nu0=[" << std::fixed << std::setprecision(4) << ctx.log_nu_min << ','
      << ctx.log_nu_max << "]\n";
  out_ << std::scientific << std::setprecision(6);
  out_ << "      vis_rows_used=" << ctx.vis_rows_used << "  random_sample_frac=" << std::fixed
      << std::setprecision(4) << ctx.random_sample_frac << "  gridding="
      << (ctx.gridding ? "on" : "off") << "  weighting=" << ctx.weighting;
  if (ctx.weighting.find("Briggs") != std::string::npos ||
      ctx.weighting.find("briggs") != std::string::npos ||
      ctx.weighting.find("robust") != std::string::npos) {
    out_ << " robust_R=" << std::fixed << std::setprecision(2) << ctx.robust_r;
  }
  out_ << '\n';
  out_ << "uv: max_uv_wavelength=" << format_sci(ctx.max_uv_wavelength)
      << "  nyquist_cell_arcsec_max=" << std::fixed << std::setprecision(6) << ctx.nyquist_cell_arcsec
      << "  model_cell_arcsec=" << ctx.cell_arcsec << '\n';
  out_ << "beam: FWHM_arcsec=" << std::fixed << std::setprecision(3) << ctx.beam_major_arcsec << 'x'
      << ctx.beam_minor_arcsec << " PA_deg=" << ctx.beam_pa_deg << "  noise_Jy_per_pixel="
      << format_sci(ctx.noise_jy_per_pixel) << "  fg_scale=" << format_sci(ctx.fg_scale) << '\n';
  out_ << "objective: phi = sum(lambda_i * value_i)\n";
  for (const auto& t : ctx.objective_terms) {
    out_ << "  " << t.name << ": lambda=" << std::fixed << std::setprecision(6) << t.lambda << "  ("
        << (t.active ? "active" : "skipped") << ")\n";
  }
  out_ << "optimize: optimizer=" << ctx.optimizer_id << "  mode=" << ctx.optimization_mode
      << "  max_iter=" << ctx.max_iter << "  ftol=" << format_sci(ctx.ftol) << " gtol="
      << format_sci(ctx.gtol) << '\n';
  out_ << "          linesearch=" << ctx.linesearch_id << "  seeder=" << ctx.seeder_id
      << "  lbfgs_m=" << ctx.lbfgs_m << "  normalize_chi2=" << (ctx.normalize_chi2 ? "yes" : "no")
      << '\n';
  out_ << std::fixed << std::setprecision(6);
  out_ << "          MINPIX=" << ctx.minpix << "  noise_cut_eff=" << format_sci(ctx.noise_cut_eff)
      << "  mask=" << ctx.mask_description << '\n';
  out_ << "gpu: devices=" << ctx.gpu_first << "  count=" << ctx.gpu_count << "  name=\""
      << ctx.gpu_name << "\"  mem_GiB=" << std::fixed << std::setprecision(1) << ctx.gpu_mem_gib
      << '\n';
  out_ << "=== end run summary ===\n\n";
  out_ << std::defaultfloat;
}

void ConsoleRunObserver::on_optimization_begin(const OptimizationBeginInfo& info) {
  if (log_level_from_runtime(runtime_) == LogLevel::Quiet) return;
  out_ << "--- Non-linear optimization ---\n";
  if (!info.mode_description.empty()) {
    out_ << info.mode_description << '\n';
  }
  out_ << "Starting " << info.optimizer_id << " (max " << info.max_iterations << " iterations).\n";
}

void ConsoleRunObserver::print_progress_bar(const IterationMetrics& ctx, double eta_s) {
  const int width = 24;
  const int max_i = std::max(ctx.max_iterations, 1);
  const int filled =
      std::min(width, std::max(0, static_cast<int>((static_cast<double>(ctx.iteration) * width) /
                                                    static_cast<double>(max_i))));
  std::string bar(static_cast<size_t>(width), ' ');
  for (int i = 0; i < filled; ++i) bar[static_cast<size_t>(i)] = '=';
  if (filled < width && filled > 0) bar[static_cast<size_t>(filled)] = '>';

  out_ << "\r[opt] [" << bar << "] " << ctx.iteration << '/' << ctx.max_iterations << "  phi="
      << format_sci(static_cast<double>(ctx.phi)) << "  chi2_w=" << format_sci(ctx.chi2_w)
      << "  reg_w=" << format_sci(ctx.reg_w) << "  " << std::fixed << std::setprecision(2)
      << ctx.wall_time_s << "s/it";
  if (ctx.iteration >= 3 && std::isfinite(eta_s)) {
    out_ << "  ETA " << format_eta_seconds(eta_s);
  }
  out_ << std::defaultfloat << std::flush;
}

void ConsoleRunObserver::on_iteration(const IterationMetrics& ctx) {
  if (log_level_from_runtime(runtime_) == LogLevel::Quiet) return;

  const ProgressMode prog =
      progress_mode_from_runtime(runtime_, isatty(fileno(stdout)) != 0);
  if (prog == ProgressMode::Off) return;
  if (!should_print_iteration(ctx.iteration, ctx.max_iterations)) return;

  recent_iter_wall_s_.push_back(ctx.wall_time_s);
  while (recent_iter_wall_s_.size() > 5) recent_iter_wall_s_.pop_front();

  double eta_s = -1.0;
  if (recent_iter_wall_s_.size() >= 2) {
    double sum = 0.0;
    for (double t : recent_iter_wall_s_) sum += t;
    const double avg = sum / static_cast<double>(recent_iter_wall_s_.size());
    eta_s = avg * static_cast<double>(ctx.max_iterations - ctx.iteration);
  }

  if (prog == ProgressMode::Bar && isatty(fileno(stdout)) != 0) {
    print_progress_bar(ctx, eta_s);
    return;
  }

  out_ << "[opt]";
  if (ctx.sub_run_total > 1) {
    out_ << " sub=" << ctx.sub_run << '/' << ctx.sub_run_total;
    if (!ctx.sub_plane_label.empty()) out_ << " plane=" << ctx.sub_plane_label;
  }
  out_ << " iter=" << ctx.iteration << '/' << ctx.max_iterations << " phi="
      << format_sci(static_cast<double>(ctx.phi)) << " chi2_w=" << format_sci(ctx.chi2_w)
      << " reg_w=" << format_sci(ctx.reg_w) << " dt_s=" << std::fixed << std::setprecision(2)
      << ctx.wall_time_s;
  if (ctx.iteration >= 3 && std::isfinite(eta_s)) out_ << " eta_s=" << static_cast<int>(eta_s + 0.5);
  out_ << '\n';
}

void ConsoleRunObserver::on_optimization_end(const OptimizationEndInfo& info) {
  if (log_level_from_runtime(runtime_) < LogLevel::Verbose) return;
  switch (info.reason) {
    case OptimizationStopReason::FunctionTolerance:
      on_diagnostic(DiagnosticLevel::Verbose,
                    "Optimizer stopped: relative change in objective below ftol" +
                        (info.detail.empty() ? "." : (" (" + info.detail + ").")));
      break;
    case OptimizationStopReason::ObjectivePlateau:
      on_diagnostic(DiagnosticLevel::Verbose,
                    "Optimizer stopped at iteration " + std::to_string(info.iteration) +
                        ": objective unchanged at float precision (plateau).");
      break;
    case OptimizationStopReason::GradientTolerance:
      on_diagnostic(DiagnosticLevel::Verbose,
                    "Optimizer stopped: max|gradient| below gtol" +
                        (info.detail.empty() ? "." : (" (" + info.detail + ").")));
      break;
    case OptimizationStopReason::MaxIterations:
      on_diagnostic(DiagnosticLevel::Verbose,
                    "Optimizer reached maximum iteration budget without meeting tolerances.");
      break;
  }
}

void ConsoleRunObserver::on_final_metrics(const FinalRunMetrics& metrics) {
  if (log_level_from_runtime(runtime_) == LogLevel::Quiet) return;
  out_ << "Optimizer finished successfully.\n\n";
  write_final_metrics(out_, metrics);
}

void ConsoleRunObserver::on_run_finished_quiet(int iteration, int max_iter, double phi,
                                               double wall_s,
                                               const std::string& output_image_path) {
  if (log_level_from_runtime(runtime_) != LogLevel::Quiet) return;
  out_ << "gpuvmem finished: iter=" << iteration << '/' << max_iter << " phi=" << format_sci(phi)
      << " wall_s=" << std::fixed << std::setprecision(1) << wall_s << '\n';
  if (!output_image_path.empty()) out_ << "output_image: " << output_image_path << '\n';
}

void ConsoleRunObserver::reset_iteration_progress() { recent_iter_wall_s_.clear(); }

}  // namespace cli
}  // namespace gpuvmem
