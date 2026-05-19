#include <gtest/gtest.h>

#include "ms/field.h"

using gpuvmem::ms::Baseline;
using gpuvmem::ms::Field;
using gpuvmem::ms::FieldMetadata;

TEST(Field, MetadataAndBaselineAccessors) {
  FieldMetadata meta;
  meta.field_id = 1;
  meta.name = "field_a";
  Field field(meta);
  EXPECT_EQ(field.field_id(), 1);
  EXPECT_EQ(field.name(), "field_a");

  Baseline& b = field.baseline(0, 1);
  EXPECT_EQ(b.antenna1(), 0);
  EXPECT_EQ(b.antenna2(), 1);
  ASSERT_NE(field.find_baseline(0, 1), nullptr);
  EXPECT_EQ(field.baselines().size(), 1u);
}
