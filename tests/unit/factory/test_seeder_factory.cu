#include <gtest/gtest.h>

#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "linesearch/linesearcher.cuh"

using gpuvmem::test::seeder_ids;
using gpuvmem::test::try_create;

class SeederFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(SeederFactoryTest, CreateReturnsNonNullWithMethodName) {
  const std::string& id = GetParam();
  std::unique_ptr<StepSizeSeeder> seeder(try_create<StepSizeSeeder, std::string>(id));
  ASSERT_NE(seeder, nullptr) << id;
  EXPECT_NE(seeder->methodName(), nullptr);
  EXPECT_GT(std::string(seeder->methodName()).size(), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredSeeders, SeederFactoryTest,
                         ::testing::ValuesIn(seeder_ids()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });
