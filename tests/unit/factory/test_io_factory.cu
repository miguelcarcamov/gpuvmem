#include <gtest/gtest.h>

#include "classes/io.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"

using gpuvmem::test::io_ids;
using gpuvmem::test::try_create;

class IoFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(IoFactoryTest, CreateReturnsNonNull) {
  const std::string& id = GetParam();
  std::unique_ptr<Io> io(try_create<Io, std::string>(id));
  ASSERT_NE(io, nullptr) << id;
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredIo, IoFactoryTest,
                         ::testing::ValuesIn(io_ids()));
