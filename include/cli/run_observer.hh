#ifndef GPUVMEM_CLI_RUN_OBSERVER_HH
#define GPUVMEM_CLI_RUN_OBSERVER_HH

#include "cli/cli_metrics.hh"

#include <string>

namespace gpuvmem {
namespace cli {

enum class DiagnosticLevel { Verbose, Debug };

/**
 * Presentation sink for a gpuvmem run. Non-owning pointer is held by Optimizer;
 * lifetime owned by Synthesizer (MFS).
 */
class IRunObserver {
 public:
  virtual ~IRunObserver() = default;

  virtual void on_diagnostic(DiagnosticLevel level, const std::string& message) = 0;
  virtual void on_run_summary(const RunSummary& summary) = 0;
  virtual void on_optimization_begin(const OptimizationBeginInfo& info) = 0;
  virtual void on_iteration(const IterationMetrics& metrics) = 0;
  virtual void on_optimization_end(const OptimizationEndInfo& info) = 0;
  virtual void on_final_metrics(const FinalRunMetrics& metrics) = 0;
  virtual void on_run_finished_quiet(int iteration, int max_iter, double phi, double wall_s,
                                     const std::string& output_image_path) = 0;
  virtual void reset_iteration_progress() = 0;
};

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_RUN_OBSERVER_HH
