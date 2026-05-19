#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace gpuvmem {
namespace test {

/**
 * Dataset-style imaging grid for tests: sky pixel scale (deg), image size N×M,
 * reference frequency nu_0 (Hz), and derived UV cell sizes deltau/deltav (wavelengths).
 *
 * Matches Image::uv_cell_u / uv_cell_v (see include/classes/image.cuh).
 */
struct ImagingGeometryParam {
  std::string label;
  long n{8};   // columns (u / RA), Image::nx
  long m{8};   // rows (v / Dec), Image::ny
  int images{1};
  double pixel_scale_ra_deg{1.0 / 3600.0};   // ~1 arcsec
  double pixel_scale_dec_deg{1.0 / 3600.0};
  float nu_0_hz{230e9f};

  static constexpr double kPi = 3.14159265358979323846;

  double pixel_scale_ra_rad() const {
    return pixel_scale_ra_deg * (kPi / 180.0);
  }
  double pixel_scale_dec_rad() const {
    return pixel_scale_dec_deg * (kPi / 180.0);
  }

  /** UV cell in u (λ): 1 / (ny * |CDELT1| rad). */
  double uv_cell_u() const {
    if (m <= 0 || pixel_scale_ra_deg == 0.0) return 0.0;
    return 1.0 / (static_cast<double>(m) * pixel_scale_ra_rad());
  }

  /** UV cell in v (λ): 1 / (nx * |CDELT2| rad). */
  double uv_cell_v() const {
    if (n <= 0 || pixel_scale_dec_deg == 0.0) return 0.0;
    return 1.0 / (static_cast<double>(n) * pixel_scale_dec_rad());
  }

  /** Wavelength in metres at nu_0. */
  double wavelength_m() const {
    if (nu_0_hz <= 0.f) return 0.0;
    return 299792458.0 / static_cast<double>(nu_0_hz);
  }
};

/** Presets inspired by E2E datasets (grid size / scale / frequency). */
inline const std::vector<ImagingGeometryParam>& dataset_geometry_presets() {
  static const std::vector<ImagingGeometryParam> kPresets = {
      {"smoke_8",
       8,
       8,
       1,
       1.0 / 3600.0,
       1.0 / 3600.0,
       230e9f},
      {"medium_64",
       64,
       64,
       1,
       0.5 / 3600.0,
       0.5 / 3600.0,
       230e9f},
      {"M87_like_256",
       256,
       256,
       1,
       0.1 / 3600.0,
       0.1 / 3600.0,
       230e9f},
      {"co65_spectral_128",
       128,
       128,
       2,
       1.0 / 3600.0,
       1.0 / 3600.0,
       100e9f},
      {"FREQ78_wide_64",
       64,
       64,
       1,
       2.0 / 3600.0,
       2.0 / 3600.0,
       345e9f},
  };
  return kPresets;
}

}  // namespace test
}  // namespace gpuvmem
