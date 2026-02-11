#ifndef GPUVMEM_UTILS_PHYSICS_UTILS_CUH
#define GPUVMEM_UTILS_PHYSICS_UTILS_CUH

#include "utils/constants.hh"

/** Speed of light in m/s. */
constexpr float LIGHTSPEED = 2.99792458e8f;

/** Convert frequency (Hz) to wavelength (m). */
__host__ __device__ float freq_to_wavelength(float freq);

/** Convert UVW coordinates from metres to lambda units. */
__host__ __device__ double metres_to_lambda(double uvw_metres, float freq);

/** Calculate Euclidean distance between two 2D points. */
__host__ __device__ float distance(float x, float y, float x0, float y0);

#endif  // GPUVMEM_UTILS_PHYSICS_UTILS_CUH
