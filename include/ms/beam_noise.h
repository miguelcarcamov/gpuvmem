#ifndef GPUVMEM_MS_BEAM_NOISE_H
#define GPUVMEM_MS_BEAM_NOISE_H

namespace gpuvmem {
namespace ms {

class MeasurementSet;

/**
 * Accumulate beam and noise contributions from one measurement set.
 * Iterates field → baseline → time_sample → vis; uses imaging_weight and
 * frequency from metadata. Outputs are accumulated (caller zeros before first
 * dataset).
 */
void computeNoiseAndBeamContribution(const MeasurementSet& ms,
                                     double* s_uu,
                                     double* s_vv,
                                     double* s_uv,
                                     float* sum_weights,
                                     int* total_visibilities);

/**
 * Compute final beam (bmaj, bmin, bpa in degrees) and noise from accumulated
 * sums (after summing over all datasets).
 */
void beamNoiseFromSums(double s_uu,
                       double s_vv,
                       double s_uv,
                       float sum_weights,
                       double* bmaj,
                       double* bmin,
                       double* bpa,
                       float* noise);

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_BEAM_NOISE_H
