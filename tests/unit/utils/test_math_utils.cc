#include <gtest/gtest.h>

#include "utils/math_utils.hh"

TEST(MathUtils, MedianOddCount) {
  std::vector<float> v = {3.f, 1.f, 2.f};
  EXPECT_FLOAT_EQ(median(v), 2.f);
}

TEST(MathUtils, MedianEvenCount) {
  std::vector<float> v = {4.f, 1.f, 3.f, 2.f};
  EXPECT_FLOAT_EQ(median(v), 2.5f);
}

TEST(MathUtils, IDivUp) {
  EXPECT_EQ(iDivUp(10, 3), 4);
  EXPECT_EQ(iDivUp(9, 3), 3);
}

TEST(MathUtils, IsPow2) {
  EXPECT_TRUE(isPow2(1u));
  EXPECT_TRUE(isPow2(8u));
  EXPECT_FALSE(isPow2(6u));
}

TEST(MathUtils, NearestPowerOf2) {
  EXPECT_EQ(NearestPowerOf2(5u), 8u);
  EXPECT_EQ(NearestPowerOf2(8u), 8u);
}
