/* Beam and noise from measurement set (dataset responsibility). */
#include "ms/beam_noise.h"
#include "ms/measurement_set.h"
#include "ms/measurement_set_metadata.h"
#include "ms/time_sample.h"

#include "MSFITSIO.cuh"

#include <cmath>

namespace gpuvmem {
namespace ms {

void computeNoiseAndBeamContribution(const MeasurementSet& ms,
                                     double* s_uu,
                                     double* s_vv,
                                     double* s_uv,
                                     float* sum_weights,
                                     int* total_visibilities) {
  const MeasurementSetMetadata& meta = ms.metadata();
  for (size_t f = 0; f < ms.num_fields(); f++) {
    const Field& field = ms.field(f);
    for (const Baseline& bl : field.baselines()) {
      for (const TimeSample& ts : bl.time_samples()) {
        const DataDescription* dd =
            meta.find_data_description(ts.data_desc_id());
        if (!dd) continue;
        const SpectralWindow* spw =
            meta.find_spectral_window(dd->spectral_window_id());
        if (!spw) continue;
        const double3 uvw = ts.uvw();
        for (const TimeSample::VisSample& vis : ts.visibilities()) {
          if (vis.chan < 0 || vis.chan >= spw->nchan()) continue;
          float nu = static_cast<float>(spw->frequency(vis.chan));
          float w = vis.imaging_weight;
          double u_lambda = metres_to_lambda(uvw.x, nu);
          double v_lambda = metres_to_lambda(uvw.y, nu);
          *s_uu += u_lambda * u_lambda * w;
          *s_vv += v_lambda * v_lambda * w;
          *s_uv += u_lambda * v_lambda * w;
          *sum_weights += w;
          (*total_visibilities)++;
        }
      }
    }
  }
}

namespace {
const double kPi = 3.14159265358979323846;
const double kRpdeg = kPi / 180.0;

void beamSizeFromSums(double s_uu, double s_vv, double s_uv,
                      double* bmaj_rad, double* bmin_rad, double* bpa_rad) {
  double uv_sq = s_uv * s_uv;
  double uu_minus_vv = s_uu - s_vv;
  double uu_plus_vv = s_uu + s_vv;
  double sqrt_in = std::sqrt((uu_minus_vv * uu_minus_vv) + 4.0 * uv_sq);
  *bmaj_rad = 1.0 / std::sqrt(2.0) / kPi / std::sqrt(uu_plus_vv - sqrt_in);
  *bmin_rad = 1.0 / std::sqrt(2.0) / kPi / std::sqrt(uu_plus_vv + sqrt_in);
  *bpa_rad = -0.5 * std::atan2(2.0 * s_uv, uu_minus_vv);
}
}  // namespace

void beamNoiseFromSums(double s_uu,
                       double s_vv,
                       double s_uv,
                       float sum_weights,
                       double* bmaj,
                       double* bmin,
                       double* bpa,
                       float* noise) {
  if (sum_weights <= 0.0f) {
    if (noise) *noise = 0.0f;
    return;
  }
  double su = s_uu / sum_weights;
  double sv = s_vv / sum_weights;
  double suv = s_uv / sum_weights;
  float variance = 1.0f / sum_weights;
  double bmaj_rad, bmin_rad, bpa_rad;
  beamSizeFromSums(su, sv, suv, &bmaj_rad, &bmin_rad, &bpa_rad);
  if (bmaj) *bmaj = bmaj_rad / kRpdeg;
  if (bmin) *bmin = bmin_rad / kRpdeg;
  if (bpa) *bpa = bpa_rad / kRpdeg;
  if (noise) *noise = 0.5f * std::sqrt(variance);
}

}  // namespace ms
}  // namespace gpuvmem
