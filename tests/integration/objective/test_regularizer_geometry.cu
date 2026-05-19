#include <gtest/gtest.h>

#include "classes/fi.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "imaging_geometry.hh"
#include "legacy_imaging_globals.hh"
#include "objective_cuda_harness.hh"

#include <sstream>
#include <tuple>

extern double deltau, deltav;
extern float nu_0;

using gpuvmem::test::dataset_geometry_presets;
using gpuvmem::test::fi_factory_light_ids;
using gpuvmem::test::ImagingGeometryParam;
using gpuvmem::test::ObjectiveCudaHarness;
using gpuvmem::test::try_create;

struct RegularizerGeometryCase {
  std::string fi_id;
  ImagingGeometryParam geometry;
};

class RegularizerGeometryTest : public ::testing::TestWithParam<RegularizerGeometryCase> {};

TEST_P(RegularizerGeometryTest, CalcFiWithDatasetGeometry) {
  const RegularizerGeometryCase param = GetParam();
  const std::string& id = param.fi_id;

  ObjectiveCudaHarness harness;
  harness.geometry = param.geometry;
  if (!harness.init_cuda()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  harness.apply_geometry();

  int image_index = 0;
  int image_to_add = 0;
  int z_index = 0;
  if (id == "TotalSquaredVariation") {
    if (harness.geometry.images < 2) {
      GTEST_SKIP() << "TSV needs image_count >= 2";
    }
    image_index = 1;
    image_to_add = 1;
    z_index = 2;
  }

  harness.init_launch_grid();
  harness.alloc_noise_mask(1.f);
  harness.alloc_uniform_image(2.f);

  EXPECT_NEAR(deltau, param.geometry.uv_cell_u(), 1e-9);
  EXPECT_NEAR(deltav, param.geometry.uv_cell_v(), 1e-9);
  EXPECT_FLOAT_EQ(nu_0, param.geometry.nu_0_hz);

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

static std::vector<RegularizerGeometryCase> regularizer_geometry_matrix() {
  std::vector<RegularizerGeometryCase> cases;
  for (const auto& geom : dataset_geometry_presets()) {
    for (const auto& fi_id : fi_factory_light_ids()) {
      if (fi_id == "TotalSquaredVariation" && geom.images < 2) continue;
      cases.push_back({fi_id, geom});
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    DatasetPixelAndFrequency, RegularizerGeometryTest,
    ::testing::ValuesIn(regularizer_geometry_matrix()),
    [](const ::testing::TestParamInfo<RegularizerGeometryCase>& info) {
      const auto& c = info.param;
      std::ostringstream os;
      os << c.fi_id << "__" << c.geometry.label;
      std::string name = os.str();
      for (char& ch : name) {
        if (ch == '-' || ch == ' ' || ch == '.') ch = '_';
      }
      return name;
    });
