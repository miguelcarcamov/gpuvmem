#include <gtest/gtest.h>

#include <omp.h>

#include "weightingscheme.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"

using gpuvmem::test::try_create;
using gpuvmem::test::weighting_ids;

class WeightingFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(WeightingFactoryTest, CreateReturnsNonNull) {
  const std::string& id = GetParam();
  std::unique_ptr<WeightingScheme> ws(try_create<WeightingScheme, std::string>(id));
  ASSERT_NE(ws, nullptr) << id;
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredWeighting, WeightingFactoryTest,
                         ::testing::ValuesIn(weighting_ids()));
