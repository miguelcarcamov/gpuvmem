#include <gtest/gtest.h>

#include "factory_catalog.hh"
#include "factory_test_utils.hh"
#include "linesearch/linesearcher.cuh"

using gpuvmem::test::linesearch_ids;
using gpuvmem::test::try_create;

class LineSearcherFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(LineSearcherFactoryTest, CreateReturnsNonNull) {
  const std::string& id = GetParam();
  std::unique_ptr<LineSearcher> ls(try_create<LineSearcher, std::string>(id));
  ASSERT_NE(ls, nullptr) << id;
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredLineSearchers, LineSearcherFactoryTest,
                         ::testing::ValuesIn(linesearch_ids()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });
