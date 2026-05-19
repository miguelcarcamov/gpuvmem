#include <gtest/gtest.h>

#include "classes/synthesizer.cuh"
#include "factory_catalog.hh"
#include "factory_test_utils.hh"

using gpuvmem::test::synthesizer_ids;
using gpuvmem::test::try_create;

class SynthesizerFactoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(SynthesizerFactoryTest, CreateReturnsNonNull) {
  const std::string& id = GetParam();
  std::unique_ptr<Synthesizer> sy(try_create<Synthesizer, std::string>(id));
  ASSERT_NE(sy, nullptr) << id;
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredSynthesizer, SynthesizerFactoryTest,
                         ::testing::ValuesIn(synthesizer_ids()));
