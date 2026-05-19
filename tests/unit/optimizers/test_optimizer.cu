#include <gtest/gtest.h>

#include "classes/optimizer.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "optimizers/lbfgs.cuh"

using gpuvmem::test::try_create;

TEST(Optimizer, IterationSetters) {
  std::unique_ptr<Optimizer> opt(try_create<Optimizer, std::string>("CG-PolakRibiere"));
  ASSERT_NE(opt, nullptr);
  opt->setTotalIterations(42);
  EXPECT_EQ(opt->getTotalIterations(), 42);
  EXPECT_EQ(opt->getCurrentIteration(), 0);
}

TEST(LBFGS, HistorySizeSetter) {
  std::unique_ptr<Optimizer> opt(try_create<Optimizer, std::string>("LBFGS"));
  auto* lbfgs = dynamic_cast<LBFGS*>(opt.get());
  ASSERT_NE(lbfgs, nullptr);
  EXPECT_EQ(lbfgs->getHistorySize(), 100);
  lbfgs->setHistorySize(15);
  EXPECT_EQ(lbfgs->getHistorySize(), 15);
}
