#include <gtest/gtest.h>

#include "classes/fi.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "objective_cuda_harness.hh"

using gpuvmem::test::fi_factory_light_ids;
using gpuvmem::test::ObjectiveCudaHarness;
using gpuvmem::test::try_create;

class RegularizerCalcFiTest : public ::testing::TestWithParam<std::string> {};

TEST_P(RegularizerCalcFiTest, CalcFiIsNonNegative) {
  const std::string& id = GetParam();
  if (id == "Chi2") {
    GTEST_SKIP() << "Chi2 requires Image/MS; covered in E2E";
  }

  ObjectiveCudaHarness harness;
  if (!harness.init_cuda()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  harness.apply_geometry();
  int image_index = 0;
  int image_to_add = 0;
  int z_index = 0;
  if (id == "TotalSquaredVariation") {
    harness.images = 2;
    image_index = 1;
    image_to_add = 1;
    z_index = 2;
  }
  harness.init_launch_grid();
  harness.alloc_noise_mask(1.f);
  harness.alloc_uniform_image(2.f);

  std::unique_ptr<Fi> term(try_create<Fi, std::string>(id));
  ASSERT_NE(term, nullptr) << id;
  term->setPrior(0.001f);

  ObjectiveFunction of = harness.make_objective();
  harness.wire_production_weights(of);

  ASSERT_TRUE(harness.add_term(of, term.get(), z_index, image_index, image_to_add, false));

  const float phi = harness.eval_phi(of);
  EXPECT_GE(phi, 0.f);

  harness.teardown_cuda();
}

INSTANTIATE_TEST_SUITE_P(EachRegisteredRegularizer, RegularizerCalcFiTest,
                         ::testing::ValuesIn(fi_factory_light_ids()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           std::string name = info.param;
                           for (char& c : name) {
                             if (c == '-' || c == ' ') c = '_';
                           }
                           return name;
                         });
