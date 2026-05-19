#include <gtest/gtest.h>

#include "ms/field.h"
#include "ms/measurement_set.h"

using gpuvmem::ms::DataDescription;
using gpuvmem::ms::Field;
using gpuvmem::ms::FieldMetadata;
using gpuvmem::ms::MeasurementSet;
using gpuvmem::ms::Polarization;
using gpuvmem::ms::SpectralWindow;

TEST(MeasurementSetIntegration, FieldUsesSharedMetadata) {
  MeasurementSet ms("test.ms");

  ms.metadata().add_spectral_window(SpectralWindow(0, {1.0e9, 1.1e9}));
  ms.metadata().add_polarization(Polarization(0, 2, {5, 6}));
  ms.metadata().add_data_description(DataDescription(0, 0, 0, 2, 2));

  FieldMetadata fmeta;
  fmeta.field_id = 3;
  fmeta.name = "3C279";
  Field& field = ms.add_field(fmeta);
  EXPECT_EQ(field.field_id(), 3);
  EXPECT_EQ(field.name(), "3C279");

  field.baseline(1, 2);
  ASSERT_NE(field.find_baseline(1, 2), nullptr);
  EXPECT_EQ(field.baselines().size(), 1u);

  EXPECT_EQ(ms.num_fields(), 1u);
  EXPECT_EQ(ms.metadata().data_description(0).nchan(), 2);
}
