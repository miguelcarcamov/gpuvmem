#include <gtest/gtest.h>

#include "classes/ckernel.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"

using gpuvmem::test::ckernel_ids;
using gpuvmem::test::try_create;

class CKernelFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(CKernelFactoryTest, CreateReturnsNonNull) {
  const std::string& id = GetParam();
  std::unique_ptr<CKernel> k(try_create<CKernel, std::string>(id));
  ASSERT_NE(k, nullptr) << id;
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredCKernels, CKernelFactoryTest,
                         ::testing::ValuesIn(ckernel_ids()));
