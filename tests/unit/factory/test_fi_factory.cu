#include <gtest/gtest.h>

#include "classes/fi.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"

using gpuvmem::test::fi_factory_light_ids;
using gpuvmem::test::try_create;

class FiFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(FiFactoryTest, CreateReturnsNonNullWithExpectedName) {
  const std::string& id = GetParam();
  std::unique_ptr<Fi> fi(try_create<Fi, std::string>(id));
  ASSERT_NE(fi, nullptr) << "missing factory id: " << id;
  EXPECT_FALSE(fi->getName().empty());
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredFi, FiFactoryTest,
                         ::testing::ValuesIn(fi_factory_light_ids()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           std::string name = info.param;
                           for (char& c : name) {
                             if (c == '-' || c == ' ') c = '_';
                           }
                           return name;
                         });
