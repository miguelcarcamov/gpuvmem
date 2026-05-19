#include <gtest/gtest.h>

#include "cli/gpuvmem_cli_config.hh"
#include "gtest_getopt_reset.hh"

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

TEST(CliParse, WarrantyExitsWithoutInputValidation) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-w"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  EXPECT_TRUE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
  EXPECT_TRUE(cfg.runtime.print_warranty);
}

TEST(CliParse, QuietAndVerboseAreMutuallyExclusive) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits",
                                           "-z", "0.001", "-q", "-v"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  EXPECT_FALSE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
  EXPECT_NE(err.str().find("mutually exclusive"), std::string::npos);
}

TEST(CliParse, ProgressModePlain) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits",
                                           "-z", "0.001", "--progress", "plain"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  ASSERT_TRUE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
  EXPECT_EQ(cfg.runtime.progress_mode, GpuvmemCliRuntimeFlags::ProgressPlain);
}

TEST(CliParse, InvalidProgressModeRejected) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits",
                                           "-z", "0.001", "--progress", "invalid"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  EXPECT_FALSE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
}

TEST(CliParse, SyntheticGridRequiresAllGeometryFlags) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-i", "in.ms", "-o", "out.ms", "-z", "0.001",
                                           "--imsize", "64,64"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  EXPECT_FALSE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
}

TEST(CliParse, LogIntervalMustBePositive) {
  gpuvmem_gtest_reset_getopt();
  const std::vector<std::string> tokens = {"gpuvmem", "-i", "in.ms", "-o", "out.ms", "-m", "m.fits",
                                           "-z", "0.001", "--log-interval", "0"};
  auto argv = argv_from_tokens(tokens);
  GpuvmemCliConfig cfg;
  std::ostringstream err;
  EXPECT_FALSE(parse_gpuvmem_cli(static_cast<int>(argv.size()), argv.data(), cfg, err));
}
