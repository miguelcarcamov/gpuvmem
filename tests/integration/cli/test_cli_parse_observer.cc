#include <gtest/gtest.h>

#include "cli/console_run_observer.hh"
#include "cli/gpuvmem_cli_config.hh"
#include "gtest_getopt_reset.hh"
#include "cli/run_observer_factory.hh"

#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<char*> argv_from_tokens(const std::vector<std::string>& tokens) {
  std::vector<char*> argv;
  argv.reserve(tokens.size());
  for (const auto& t : tokens) {
    argv.push_back(const_cast<char*>(t.c_str()));
  }
  return argv;
}

}  // namespace

TEST(CliParseObserverIntegration, PlainProgressParsedAndObserverCreated) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {
      "gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits",
      "-z",    "0.001", "--progress", "plain"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  ASSERT_TRUE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
  EXPECT_EQ(cfg.runtime.progress_mode, GpuvmemCliRuntimeFlags::ProgressPlain);

  auto observer = gpuvmem::cli::create_run_observer(cfg.runtime);
  ASSERT_NE(observer, nullptr);

  std::ostringstream out;
  gpuvmem::cli::ConsoleRunObserver console(cfg.runtime, out);
  gpuvmem::cli::IterationMetrics m;
  m.iteration = 1;
  m.max_iterations = 10;
  m.phi = 1.f;
  m.chi2_w = 0.8f;
  m.reg_w = 0.2f;
  console.on_iteration(m);
  EXPECT_NE(out.str().find("iter=1/10"), std::string::npos);
}

// Re-test with captured stream via factory-built observer
TEST(CliParseObserverIntegration, ParsedQuietSkipsSummary) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {
      "gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits", "-z", "0.001", "-q"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  ASSERT_TRUE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
  EXPECT_TRUE(cfg.runtime.quiet);

  std::ostringstream captured;
  gpuvmem::cli::ConsoleRunObserver observer(cfg.runtime, captured);
  observer.on_run_summary(gpuvmem::cli::RunSummary{});
  EXPECT_TRUE(captured.str().empty());
}
