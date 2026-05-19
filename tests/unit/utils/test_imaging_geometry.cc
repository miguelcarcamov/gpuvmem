#include <gtest/gtest.h>

#include "imaging_geometry.hh"

using gpuvmem::test::dataset_geometry_presets;
using gpuvmem::test::ImagingGeometryParam;

TEST(ImagingGeometry, UvCellMatchesImageFormula) {
  ImagingGeometryParam g;
  g.n = 64;
  g.m = 64;
  g.pixel_scale_ra_deg = 0.5 / 3600.0;
  g.pixel_scale_dec_deg = 0.5 / 3600.0;

  const double expected_u =
      1.0 / (static_cast<double>(g.m) * g.pixel_scale_ra_deg * (g.kPi / 180.0));
  const double expected_v =
      1.0 / (static_cast<double>(g.n) * g.pixel_scale_dec_deg * (g.kPi / 180.0));

  EXPECT_NEAR(g.uv_cell_u(), expected_u, 1e-12);
  EXPECT_NEAR(g.uv_cell_v(), expected_v, 1e-12);
}

TEST(ImagingGeometry, FinerPixelsGiveLargerUvCells) {
  ImagingGeometryParam coarse;
  coarse.n = 64;
  coarse.m = 64;
  coarse.pixel_scale_ra_deg = 2.0 / 3600.0;
  coarse.pixel_scale_dec_deg = 2.0 / 3600.0;

  ImagingGeometryParam fine = coarse;
  fine.pixel_scale_ra_deg = 0.5 / 3600.0;
  fine.pixel_scale_dec_deg = 0.5 / 3600.0;

  EXPECT_GT(fine.uv_cell_u(), coarse.uv_cell_u());
  EXPECT_GT(fine.uv_cell_v(), coarse.uv_cell_v());
}

TEST(ImagingGeometry, DatasetPresetsHavePositiveFrequencyAndCells) {
  for (const auto& preset : dataset_geometry_presets()) {
    EXPECT_GT(preset.n, 0L) << preset.label;
    EXPECT_GT(preset.m, 0L) << preset.label;
    EXPECT_GT(preset.nu_0_hz, 0.f) << preset.label;
    EXPECT_GT(preset.uv_cell_u(), 0.0) << preset.label;
    EXPECT_GT(preset.uv_cell_v(), 0.0) << preset.label;
  }
}
