#ifndef ERRORS_HOST_CUH
#define ERRORS_HOST_CUH

#include "classes/image.cuh"

// Error calculation host functions
__host__ void calculateErrors(Image* image, float fg_scale);
__host__ void precomputeNeff(bool normalize);

#endif  // ERRORS_HOST_CUH
