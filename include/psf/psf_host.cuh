#ifndef PSF_HOST_CUH
#define PSF_HOST_CUH

// SYNTHESIZED/DIRTY BEAM (PSF) MODULE
// Handles point spread function (PSF) calculations from uv-coverage
// that characterize image resolution (bmaj, bmin, bpa).
// This is distinct from the primary beam (antenna response pattern) in beam/ module.

#include "framework.cuh"
#include <vector>

// Synthesized/Dirty Beam (PSF) Functions
// These functions calculate the point spread function (PSF) of the interferometer
// from uv-coverage, characterized by bmaj, bmin, bpa parameters.
// This is distinct from the primary beam (antenna response pattern).

// Calculate synthesized beam shape parameters from uv-coverage
// Computes weighted second moments: s_uu, s_vv, s_uv
__host__ void calc_sBeam(std::vector<double3> uvw,
                         std::vector<float> weight,
                         float nu,
                         double* s_uu,
                         double* s_vv,
                         double* s_uv);

// Calculate synthesized beam size parameters (bmaj, bmin, bpa) from shape parameters
// Returns: (major_axis_rad, minor_axis_rad, position_angle_rad)
__host__ double3 calc_beamSize(double s_uu, double s_vv, double s_uv);

// Calculate noise and synthesized beam parameters from measurement sets
// Computes bmaj, bmin, bpa (synthesized beam) and noise from uv-coverage
__host__ float calculateNoiseAndBeam(std::vector<MSDataset>& datasets,
                                     int* total_visibilities,
                                     int blockSizeV,
                                     double* bmaj,
                                     double* bmin,
                                     double* bpa,
                                     float* noise);

#endif  // PSF_HOST_CUH
