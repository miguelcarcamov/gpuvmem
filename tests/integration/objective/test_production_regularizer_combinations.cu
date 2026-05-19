#include <gtest/gtest.h>

#include "classes/fi.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "objective_cuda_harness.hh"

#include <cmath>
#include <sstream>
#include <vector>

using gpuvmem::test::ObjectiveCudaHarness;
using gpuvmem::test::production_fi_ids;
using gpuvmem::test::try_create;

struct ProductionRegComboParam {
  unsigned mask;  // bit i -> production_fi_ids[i+1] (Entropy, L1, TSV)
  std::string label;
};

static std::vector<ProductionRegComboParam> production_regularizer_combinations() {
  std::vector<ProductionRegComboParam> params;
  const unsigned nregs = 3u;
  for (unsigned mask = 0; mask < (1u << nregs); ++mask) {
    ProductionRegComboParam p;
    p.mask = mask;
    std::ostringstream os;
    os << "regs";
    if (mask & 1u) os << "_entropy";
    if (mask & 2u) os << "_l1";
    if (mask & 4u) os << "_tsv";
    if (mask == 0u) os << "_none";
    p.label = os.str();
    params.push_back(p);
  }
  return params;
}

class ProductionRegularizerComboTest
    : public ::testing::TestWithParam<ProductionRegComboParam> {};

TEST_P(ProductionRegularizerComboTest, WiresProductionStyleRegularizers) {
  const ProductionRegComboParam param = GetParam();
  const auto& ids = production_fi_ids();
  ASSERT_EQ(ids.size(), 4u);

  ObjectiveCudaHarness harness;
  if (!harness.init_cuda()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  harness.apply_geometry();
  if (param.mask & 4u) harness.images = 2;
  harness.init_launch_grid();
  harness.alloc_noise_mask(1.f);
  harness.alloc_uniform_image(2.f);

  ObjectiveFunction of = harness.make_objective();
  harness.wire_production_weights(of);

  const std::vector<std::pair<unsigned, int>> reg_specs = {
      {1u, 0}, {2u, 1}, {4u, 2}};  // mask bit, z_index
  std::vector<std::unique_ptr<Fi>> owned_terms;
  int expected_terms = 0;
  for (const auto& spec : reg_specs) {
    if ((param.mask & spec.first) == 0u) continue;
    const std::string& reg_id = ids[1 + spec.second];
    std::unique_ptr<Fi> term(try_create<Fi, std::string>(reg_id));
    ASSERT_NE(term, nullptr) << reg_id;
    term->setPrior(0.001f);
    int image_index = 0;
    int image_to_add = 0;
    if (reg_id == "TotalSquaredVariation") {
      image_index = 1;
      image_to_add = 1;
    }
    if (harness.add_term(of, term.get(), spec.second, image_index, image_to_add, false)) {
      ++expected_terms;
    }
    owned_terms.push_back(std::move(term));
  }
  EXPECT_EQ(of.getFi().size(), static_cast<size_t>(expected_terms));

  if (expected_terms > 0) {
    const float phi = harness.eval_phi(of);
    EXPECT_GE(phi, 0.f);
    EXPECT_TRUE(std::isfinite(phi));
  }

  harness.teardown_cuda();
}

INSTANTIATE_TEST_SUITE_P(ProductionPowerSet, ProductionRegularizerComboTest,
                         ::testing::ValuesIn(production_regularizer_combinations()),
                         [](const ::testing::TestParamInfo<ProductionRegComboParam>& info) {
                           return info.param.label;
                         });
