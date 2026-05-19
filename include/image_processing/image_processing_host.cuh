#ifndef IMAGE_PROCESSING_HOST_CUH
#define IMAGE_PROCESSING_HOST_CUH

#include <cufft.h>

__host__ void normalizeImage(float* image, float normalization_factor);

/** Clip each image plane by noise (Stokes or single-plane; image_count != 2). */
__host__ void linkClipStokesWNoise(float* I, int nplanes);

/** Copy single-plane I to device_I_nu (real = I, imag = 0). For Stokes/single-image (image_count != 2). */
__host__ void linkCopyItoInu(cufftComplex* image, float* I);

#endif  // IMAGE_PROCESSING_HOST_CUH
