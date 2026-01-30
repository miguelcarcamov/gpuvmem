/* Primary beam calculation: single and multiple frequency (host). */
#include "ms/primary_beam.h"

#include <boost/math/special_functions/bessel.hpp>

#include <cmath>

namespace gpuvmem {
namespace ms {

namespace {

constexpr double PI_D = 3.141592653589793;

float airy_disk_beam(double angular_distance_rad,
                     double lambda,
                     float antenna_diameter,
                     float pb_factor) {
  if (antenna_diameter <= 0.f || lambda <= 0.) return 1.f;
  if (angular_distance_rad == 0.) return 1.f;
  double bessel_arg =
      PI_D * angular_distance_rad * antenna_diameter / lambda *
      (static_cast<double>(RZ_PB) / static_cast<double>(pb_factor));
  double j1_val = boost::math::cyl_bessel_j(1, bessel_arg);
  double atten = 4.0 * (j1_val / bessel_arg) * (j1_val / bessel_arg);
  return static_cast<float>(atten);
}

float gaussian_beam(double angular_distance_rad,
                   double lambda,
                   float antenna_diameter,
                   float pb_factor) {
  if (antenna_diameter <= 0.f || lambda <= 0.) return 1.f;
  double fwhm = pb_factor * lambda / antenna_diameter;
  const double c = 4.0 * std::log(2.0);
  double r = angular_distance_rad / fwhm;
  return static_cast<float>(std::exp(-c * r * r));
}

}  // namespace

float primary_beam_value(float antenna_diameter,
                        float pb_factor,
                        float pb_cutoff,
                        PrimaryBeamType type,
                        double freq_hz,
                        double angular_distance_rad) {
  if (angular_distance_rad > static_cast<double>(pb_cutoff)) return 0.f;
  if (freq_hz <= 0.) return 0.f;
  double lambda = LIGHTSPEED_PB / freq_hz;
  if (type == PrimaryBeamType::AiryDisk)
    return airy_disk_beam(angular_distance_rad, lambda, antenna_diameter,
                         pb_factor);
  return gaussian_beam(angular_distance_rad, lambda, antenna_diameter,
                      pb_factor);
}

void primary_beam_values(float antenna_diameter,
                         float pb_factor,
                         float pb_cutoff,
                         PrimaryBeamType type,
                         const double* freq_hz,
                         size_t n_freq,
                         double angular_distance_rad,
                         float* out) {
  for (size_t i = 0; i < n_freq; i++)
    out[i] = primary_beam_value(antenna_diameter, pb_factor, pb_cutoff, type,
                                freq_hz[i], angular_distance_rad);
}

std::vector<float> primary_beam_values(const Antenna& ant,
                                       const std::vector<double>& freq_hz,
                                       double angular_distance_rad) {
  std::vector<float> out(freq_hz.size());
  primary_beam_values(ant.antenna_diameter, ant.pb_factor, ant.pb_cutoff,
                      ant.primary_beam, freq_hz.data(), freq_hz.size(),
                      angular_distance_rad, out.data());
  return out;
}

}  // namespace ms
}  // namespace gpuvmem
