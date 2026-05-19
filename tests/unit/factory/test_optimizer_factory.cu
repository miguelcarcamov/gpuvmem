#include <gtest/gtest.h>

#include "classes/optimizer.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "optimizers/lbfgs.cuh"

using gpuvmem::test::optimizer_ids;
using gpuvmem::test::try_create;

class OptimizerFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(OptimizerFactoryTest, CreateAndConfigure) {
  const std::string& id = GetParam();
  std::unique_ptr<Optimizer> opt(try_create<Optimizer, std::string>(id));
  ASSERT_NE(opt, nullptr) << id;
  if (id == "LBFGS") {
    auto* lbfgs = dynamic_cast<LBFGS*>(opt.get());
    ASSERT_NE(lbfgs, nullptr);
    lbfgs->setHistorySize(7);
    EXPECT_EQ(lbfgs->getHistorySize(), 7);
  }
  opt->setFTol(1e-4f);
  opt->setGTol(1e-5f);
  EXPECT_FLOAT_EQ(opt->getFtol(), 1e-4f);
  EXPECT_FLOAT_EQ(opt->getGtol(), 1e-5f);
  opt->setTotalIterations(11);
  EXPECT_EQ(opt->getTotalIterations(), 11);
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredOptimizers, OptimizerFactoryTest,
                         ::testing::ValuesIn(optimizer_ids()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           std::string name = info.param;
                           for (char& c : name) {
                             if (c == '-' || c == ' ') c = '_';
                           }
                           return name;
                         });
