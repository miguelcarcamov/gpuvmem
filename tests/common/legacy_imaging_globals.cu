#include "legacy_imaging_globals.hh"

double deltau = 1.0;
double deltav = 1.0;
double DELTAX = 1.0;
double DELTAY = 1.0;
float nu_0 = 230e9f;

namespace gpuvmem {
namespace test {

void apply_legacy_imaging_globals(const ImagingGeometryParam& geom) {
  deltau = geom.uv_cell_u();
  deltav = geom.uv_cell_v();
  DELTAX = geom.pixel_scale_ra_deg;
  DELTAY = geom.pixel_scale_dec_deg;
  nu_0 = geom.nu_0_hz;
}

}  // namespace test
}  // namespace gpuvmem
