#ifndef GPUVMEM_MS_PRIMARY_BEAM_H
#define GPUVMEM_MS_PRIMARY_BEAM_H

#include "ms/metadata.h"

#include <cmath>
#include <vector>

namespace gpuvmem {
namespace ms {

/** Speed of light [m/s]. */
constexpr float LIGHTSPEED_PB = 2.99792458e8f;
/** First zero of J1 / pi (Airy first null at ~1.22 lambda/D). */
constexpr float RZ_PB = 1.2196698912665045f;

/**
 * Primary beam value for one antenna at one frequency and one angular distance.
 * angular_distance_rad: offset from phase center [rad].
 * Returns attenuation (0..1); 0 if beyond pb_cutoff.
 */
float primary_beam_value(float antenna_diameter,
                        float pb_factor,
                        float pb_cutoff,
                        PrimaryBeamType type,
                        double freq_hz,
                        double angular_distance_rad);

/**
 * Same as primary_beam_value but using Antenna struct.
 */
inline float primary_beam_value(const Antenna& ant,
                                double freq_hz,
                                double angular_distance_rad) {
  return primary_beam_value(ant.antenna_diameter, ant.pb_factor, ant.pb_cutoff,
                            ant.primary_beam, freq_hz, angular_distance_rad);
}

/**
 * Primary beam values for one antenna at multiple frequencies, same angular distance.
 * Fills out[i] = primary_beam_value(..., freq_hz[i], angular_distance_rad).
 */
void primary_beam_values(float antenna_diameter,
                        float pb_factor,
                        float pb_cutoff,
                        PrimaryBeamType type,
                        const double* freq_hz,
                        size_t n_freq,
                        double angular_distance_rad,
                        float* out);

/**
 * Overload: frequencies from vector, returns vector of attenuations.
 */
std::vector<float> primary_beam_values(const Antenna& ant,
                                        const std::vector<double>& freq_hz,
                                        double angular_distance_rad);

/**
 * Baseline primary beam (geometric mean of two antenna PBs) at one frequency.
 * Often used as sqrt(pb_i * pb_j) for voltage.
 */
inline float primary_beam_baseline(const Antenna& ant1,
                                  const Antenna& ant2,
                                  double freq_hz,
                                  double angular_distance_rad) {
  float p1 = primary_beam_value(ant1, freq_hz, angular_distance_rad);
  float p2 = primary_beam_value(ant2, freq_hz, angular_distance_rad);
  return std::sqrt(p1 * p2);
}

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_PRIMARY_BEAM_H
