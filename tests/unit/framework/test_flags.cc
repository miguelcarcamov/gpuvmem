#include <gtest/gtest.h>

#include "flags.cuh"
#include "gtest_getopt_reset.hh"

TEST(Flags, ParseLongOption) {
  gpuvmem_gtest_reset_getopt();
  int value = 0;
  Flags flags;
  flags.Var(value, 'n', "number", 5, "test integer");
  char arg0[] = "gpuvmem";
  char arg1[] = "--number=42";
  char* argv[] = {arg0, arg1};
  ASSERT_TRUE(flags.Parse(2, argv));
  EXPECT_EQ(value, 42);
}

TEST(Flags, BoolFlagSetsTrue) {
  gpuvmem_gtest_reset_getopt();
  bool enabled = false;
  Flags flags;
  flags.Bool(enabled, 'v', "verbose", "verbose output");
  char arg0[] = "gpuvmem";
  char arg1[] = "-v";
  char* argv[] = {arg0, arg1};
  ASSERT_TRUE(flags.Parse(2, argv));
  EXPECT_TRUE(enabled);
}

TEST(Flags, UnknownOptionFails) {
  gpuvmem_gtest_reset_getopt();
  int value = 0;
  Flags flags;
  flags.Var(value, 'n', "number", 0, "test integer");
  char arg0[] = "gpuvmem";
  char arg1[] = "--unknown";
  char* argv[] = {arg0, arg1};
  EXPECT_FALSE(flags.Parse(2, argv));
}
