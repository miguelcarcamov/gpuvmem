#include <gtest/gtest.h>

#include "classes/image.cuh"
#include "classes/optimizer.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "gtest_param_name.hh"
#include "linesearch/linesearcher.cuh"
#include "objective_cuda_harness.hh"
#include "projection/projection.hh"
#include "test_image_linesearch.hh"

#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

using gpuvmem::test::gtest_sanitize_param_name;
using gpuvmem::test::make_linesearch_image_maps;
using gpuvmem::test::optimizer_ids;
using gpuvmem::test::ObjectiveCudaHarness;
using gpuvmem::test::try_create;
using gpuvmem::test::wire_image_for_linesearch;

struct OptimizerObjectiveParam {
  std::string optimizer;
  std::string label;
};

static std::vector<OptimizerObjectiveParam> optimizer_objective_smoke_cases() {
  std::vector<OptimizerObjectiveParam> params;
  for (const auto& opt : optimizer_ids()) {
    OptimizerObjectiveParam p{opt, ""};
    std::ostringstream os;
    os << opt << "__Fixed__L2";
    p.label = os.str();
    params.push_back(p);
  }
  return params;
}

class OptimizerObjectiveSmokeTest
    : public ::testing::TestWithParam<OptimizerObjectiveParam> {};

TEST_P(OptimizerObjectiveSmokeTest, RunsTwoIterationsOnL2Prior) {
  const OptimizerObjectiveParam& p = GetParam();

  ObjectiveCudaHarness harness;
  if (!harness.init_cuda()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  harness.apply_geometry();
  harness.init_launch_grid();
  harness.alloc_noise_mask(1.f);
  harness.alloc_uniform_image(2.f);

  std::unique_ptr<Fi> l2(try_create<Fi, std::string>("L2ConstantPrior"));
  ASSERT_NE(l2, nullptr);
  l2->setPrior(0.01f);

  ObjectiveFunction of = harness.make_objective();
  float w[] = {0.f, 0.f, 0.f, 1.f};
  of.setRegularizationWeights(w, 4);
  ASSERT_TRUE(harness.add_term(of, l2.get(), 3, 0, 0, false));

  Image image(harness.device_image, harness.images, harness.m, harness.n);
  std::vector<imageMap> image_maps = make_linesearch_image_maps(harness.images);
  wire_image_for_linesearch(image, image_maps);

  std::unique_ptr<Optimizer> opt(try_create<Optimizer, std::string>(p.optimizer));
  std::unique_ptr<LineSearcher> ls(try_create<LineSearcher, std::string>("Fixed"));
  ASSERT_NE(opt, nullptr) << p.optimizer;
  ASSERT_NE(ls, nullptr);
  ls->setProjection(std::make_unique<NoProjection>());
  opt->setLineSearcher(std::move(ls));
  opt->setTotalIterations(1);
  opt->setFTol(0.f);
  opt->setGTol(0.f);
  opt->setObjectiveFunction(&of);
  opt->setImage(&image);

  const float phi0 = harness.eval_phi(of);
  ASSERT_TRUE(std::isfinite(phi0));
  ASSERT_GT(phi0, 0.f);

  opt->optimize();

  const float phi1 = harness.eval_phi(of);
  EXPECT_TRUE(std::isfinite(phi1));
  EXPECT_GE(phi1, 0.f);

  harness.teardown_cuda();
}

INSTANTIATE_TEST_SUITE_P(EachOptimizer, OptimizerObjectiveSmokeTest,
                         ::testing::ValuesIn(optimizer_objective_smoke_cases()),
                         [](const ::testing::TestParamInfo<OptimizerObjectiveParam>& info) {
                           return gtest_sanitize_param_name(info.param.label);
                         });
