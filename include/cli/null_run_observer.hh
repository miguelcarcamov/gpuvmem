#ifndef GPUVMEM_CLI_NULL_RUN_OBSERVER_HH
#define GPUVMEM_CLI_NULL_RUN_OBSERVER_HH

#include "cli/run_observer.hh"

namespace gpuvmem {
namespace cli {

/** No-op observer (tests, or explicit silencing of presentation hooks). */
class NullRunObserver : public IRunObserver {
 public:
  void on_diagnostic(DiagnosticLevel, const std::string&) override {}
  void on_run_summary(const RunSummary&) override {}
  void on_optimization_begin(const OptimizationBeginInfo&) override {}
  void on_iteration(const IterationMetrics&) override {}
  void on_optimization_end(const OptimizationEndInfo&) override {}
  void on_final_metrics(const FinalRunMetrics&) override {}
  void on_run_finished_quiet(int, int, double, double, const std::string&) override {}
  void reset_iteration_progress() override {}
};

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_NULL_RUN_OBSERVER_HH
