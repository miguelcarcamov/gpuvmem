#ifndef GPUVMEM_CLI_CONSOLE_RUN_OBSERVER_HH
#define GPUVMEM_CLI_CONSOLE_RUN_OBSERVER_HH

#include "cli/gpuvmem_cli_config.hh"
#include "cli/run_observer.hh"

#include <deque>
#include <iostream>
#include <iosfwd>
#include <ostream>

namespace gpuvmem {
namespace cli {

enum class LogLevel { Quiet, Normal, Verbose, Debug };
enum class ProgressMode { Auto, Bar, Plain, Off };

LogLevel log_level_from_runtime(const GpuvmemCliRuntimeFlags& runtime);
ProgressMode progress_mode_from_runtime(const GpuvmemCliRuntimeFlags& runtime, bool is_tty);

/** Writes human-readable progress to an ostream (default stdout). */
class ConsoleRunObserver : public IRunObserver {
 public:
  explicit ConsoleRunObserver(const GpuvmemCliRuntimeFlags& runtime,
                              std::ostream& out = std::cout);

  void on_diagnostic(DiagnosticLevel level, const std::string& message) override;
  void on_run_summary(const RunSummary& summary) override;
  void on_optimization_begin(const OptimizationBeginInfo& info) override;
  void on_iteration(const IterationMetrics& metrics) override;
  void on_optimization_end(const OptimizationEndInfo& info) override;
  void on_final_metrics(const FinalRunMetrics& metrics) override;
  void on_run_finished_quiet(int iteration, int max_iter, double phi, double wall_s,
                             const std::string& output_image_path) override;
  void reset_iteration_progress() override;

 private:
  GpuvmemCliRuntimeFlags runtime_;
  std::ostream& out_;
  std::deque<double> recent_iter_wall_s_;

  bool should_print_iteration(int iteration, int max_iterations) const;
  void print_progress_bar(const IterationMetrics& metrics, double eta_s);
};

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_CONSOLE_RUN_OBSERVER_HH
