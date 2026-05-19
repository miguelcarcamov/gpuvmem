#include <gtest/gtest.h>

#include "classes/optimizer.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "gtest_param_name.hh"
#include "linesearch/linesearcher.cuh"

#include <sstream>
#include <tuple>
#include <vector>

using gpuvmem::test::gtest_sanitize_param_name;
using gpuvmem::test::linesearch_ids;
using gpuvmem::test::optimizer_ids;
using gpuvmem::test::seeder_ids;
using gpuvmem::test::try_create;

struct OptimizerStackParam {
  std::string optimizer;
  std::string linesearch;
  std::string seeder;  // empty => no seeder (mirrors CLI without -B)
  std::string label;
};

static std::vector<OptimizerStackParam> full_optimizer_stack_matrix() {
  std::vector<OptimizerStackParam> params;
  for (const auto& opt : optimizer_ids()) {
    for (const auto& ls : linesearch_ids()) {
      // Without seeder
      {
        OptimizerStackParam p{opt, ls, "", ""};
        std::ostringstream os;
        os << opt << "__" << ls << "__no_seeder";
        p.label = os.str();
        params.push_back(p);
      }
      for (const auto& se : seeder_ids()) {
        OptimizerStackParam p{opt, ls, se, ""};
        std::ostringstream os;
        os << opt << "__" << ls << "__" << se;
        p.label = os.str();
        params.push_back(p);
      }
    }
  }
  return params;
}

class OptimizerStackMatrixTest : public ::testing::TestWithParam<OptimizerStackParam> {};

TEST_P(OptimizerStackMatrixTest, OptimizerAcceptsLineSearcherAndSeeder) {
  const OptimizerStackParam& p = GetParam();
  std::unique_ptr<Optimizer> opt(try_create<Optimizer, std::string>(p.optimizer));
  std::unique_ptr<LineSearcher> ls(try_create<LineSearcher, std::string>(p.linesearch));
  ASSERT_NE(opt, nullptr) << p.optimizer;
  ASSERT_NE(ls, nullptr) << p.linesearch;

  if (!p.seeder.empty()) {
    std::unique_ptr<StepSizeSeeder> seeder(
        try_create<StepSizeSeeder, std::string>(p.seeder));
    ASSERT_NE(seeder, nullptr) << p.seeder;
    ls->setStepSizeSeeder(std::move(seeder));
  }
  opt->setLineSearcher(std::move(ls));
  opt->setTotalIterations(3);
  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(FullMatrix, OptimizerStackMatrixTest,
                         ::testing::ValuesIn(full_optimizer_stack_matrix()),
                         [](const ::testing::TestParamInfo<OptimizerStackParam>& info) {
                           return gtest_sanitize_param_name(info.param.label);
                         });
