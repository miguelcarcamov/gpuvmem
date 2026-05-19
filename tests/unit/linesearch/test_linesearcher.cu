#include <gtest/gtest.h>

#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "linesearch/linesearcher.cuh"

using gpuvmem::test::linesearch_ids;
using gpuvmem::test::seeder_ids;
using gpuvmem::test::try_create;

TEST(LineSearcher, AttachSeeder) {
  std::unique_ptr<LineSearcher> ls(try_create<LineSearcher, std::string>("Brent"));
  std::unique_ptr<StepSizeSeeder> seeder(
      try_create<StepSizeSeeder, std::string>("BBMin1Seeder"));
  ASSERT_NE(ls, nullptr);
  ASSERT_NE(seeder, nullptr);
  ls->setStepSizeSeeder(std::move(seeder));
  SUCCEED();
}

class LineSearcherSeederPairTest
    : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(LineSearcherSeederPairTest, EveryLineSearcherAcceptsEverySeeder) {
  const auto& ls_id = GetParam().first;
  const auto& seeder_id = GetParam().second;
  std::unique_ptr<LineSearcher> ls(try_create<LineSearcher, std::string>(ls_id));
  std::unique_ptr<StepSizeSeeder> seeder(
      try_create<StepSizeSeeder, std::string>(seeder_id));
  ASSERT_NE(ls, nullptr) << ls_id;
  ASSERT_NE(seeder, nullptr) << seeder_id;
  ls->setStepSizeSeeder(std::move(seeder));
}

static std::vector<std::pair<std::string, std::string>> line_search_seeder_pairs() {
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& ls : linesearch_ids()) {
    for (const auto& se : seeder_ids()) {
      out.emplace_back(ls, se);
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(AllPairs, LineSearcherSeederPairTest,
                         ::testing::ValuesIn(line_search_seeder_pairs()));
