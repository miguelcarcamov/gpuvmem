#include <gtest/gtest.h>

#include "ms/measurement_set_metadata.h"

using gpuvmem::ms::DataDescription;
using gpuvmem::ms::MeasurementSetMetadata;
using gpuvmem::ms::Polarization;
using gpuvmem::ms::SpectralWindow;

TEST(MeasurementSetMetadata, SpectralWindowLookup) {
  MeasurementSetMetadata meta;
  meta.add_spectral_window(SpectralWindow(0, {1.0e9, 1.1e9, 1.2e9}));
  ASSERT_NE(meta.find_spectral_window(0), nullptr);
  EXPECT_EQ(meta.spectral_window(0).nchan(), 3);
  EXPECT_THROW(meta.spectral_window(99), std::out_of_range);
}

TEST(MeasurementSetMetadata, DataDescriptionLinksSpwAndPol) {
  MeasurementSetMetadata meta;
  meta.add_spectral_window(SpectralWindow(1, {2.0e9}));
  meta.add_polarization(Polarization(2, 2, {5, 6}));
  meta.add_data_description(DataDescription(10, 1, 2, 1, 2));

  const DataDescription& dd = meta.data_description(10);
  EXPECT_EQ(dd.spectral_window_id(), 1);
  EXPECT_EQ(dd.polarization_id(), 2);
  EXPECT_EQ(dd.npol(), 2);
}

TEST(MeasurementSetMetadata, FindReturnsNullForMissingId) {
  MeasurementSetMetadata meta;
  EXPECT_EQ(meta.find_data_description(42), nullptr);
}
