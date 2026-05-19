#include <gtest/gtest.h>

#include "cli/cli_metrics.hh"
#include "cli/console_run_observer.hh"
#include "cli/gpuvmem_cli_config.hh"
#include "cli/null_run_observer.hh"

#include <sstream>
#include <string>

namespace gpuvmem {
namespace cli {
namespace {

TEST(CliObserver, IterationLinePlainContainsMetrics) {
  GpuvmemCliRuntimeFlags rf;
  rf.progress_mode = GpuvmemCliRuntimeFlags::ProgressPlain;
  std::ostringstream out;
  ConsoleRunObserver observer(rf, out);

  IterationMetrics m;
  m.iteration = 3;
  m.max_iterations = 100;
  m.phi = 0.034f;
  m.chi2_w = 0.029f;
  m.reg_w = 0.005f;
  m.wall_time_s = 1.5;

  observer.on_iteration(m);
  const std::string s = out.str();
  EXPECT_NE(s.find("iter=3/100"), std::string::npos);
  EXPECT_NE(s.find("chi2_w="), std::string::npos);
  EXPECT_NE(s.find("reg_w="), std::string::npos);
  EXPECT_NE(s.find("phi="), std::string::npos);
}

TEST(CliObserver, RunSummaryHeaderInNormalMode) {
  GpuvmemCliRuntimeFlags rf;
  std::ostringstream out;
  ConsoleRunObserver observer(rf, out);

  RunSummary summary;
  summary.n_ms = 1;
  summary.input_ms = {"in.ms"};
  summary.output_ms = {"out.ms"};
  summary.model_description = "FITS mod.fits";
  summary.planes_description = "1 single_plane";
  summary.optimizer_id = "LBFGS";

  observer.on_run_summary(summary);
  const std::string s = out.str();
  EXPECT_NE(s.find("=== gpuvmem run summary ==="), std::string::npos);
  EXPECT_NE(s.find("in.ms"), std::string::npos);
}

TEST(CliObserver, QuietSkipsRunSummary) {
  GpuvmemCliRuntimeFlags rf;
  rf.quiet = true;
  std::ostringstream out;
  ConsoleRunObserver observer(rf, out);

  RunSummary summary;
  summary.n_ms = 1;
  observer.on_run_summary(summary);
  EXPECT_TRUE(out.str().empty());
}

TEST(CliObserver, NullObserverProducesNoOutput) {
  NullRunObserver observer;
  std::ostringstream out;
  (void)out;
  IterationMetrics m;
  m.iteration = 1;
  observer.on_iteration(m);
  observer.on_run_summary(RunSummary{});
  SUCCEED();
}

}  // namespace
}  // namespace cli
}  // namespace gpuvmem
