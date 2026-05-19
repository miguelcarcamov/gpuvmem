#ifndef ERRORS_KERNELS_CUH
#define ERRORS_KERNELS_CUH

#include <cufft.h>

// Error calculation kernels for Fisher information matrix computation
// These kernels compute variance and covariance for I_nu_0 and alpha parameters

__global__ void I_nu_0_Noise(float* noise_I,
                             float* images,
                             float* noise,
                             float noise_cut,
                             float nu,
                             float nu_0,
                             float* w,
                             float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             float sum_weights,
                             float fg_scale,
                             long N,
                             long M,
                             int primary_beam);

__global__ void alpha_Noise(float* noise_I,
                            float* images,
                            float nu,
                            float nu_0,
                            float* noise,
                            float noise_cut,
                            double DELTAX,
                            double DELTAY,
                            float xobs,
                            float yobs,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            float sum_weights,
                            float fg_scale,
                            long N,
                            long M,
                            int primary_beam);

__global__ void covariance_Noise(float* noise_cov,
                                 float* images,
                                 float nu,
                                 float nu_0,
                                 float* noise,
                                 float noise_cut,
                                 double DELTAX,
                                 double DELTAY,
                                 float xobs,
                                 float yobs,
                                 float antenna_diameter,
                                 float pb_factor,
                                 float pb_cutoff,
                                 float sum_weights,
                                 float fg_scale,
                                 long N,
                                 long M,
                                 int primary_beam);

__global__ void noise_reduction(float* noise_I, long N, long M);

__global__ void stokes_Noise(float* noise_I,
                             int pol_plane,
                             float nu,
                             float* noise,
                             float noise_cut,
                             float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             float sum_weights,
                             float fg_scale,
                             long N,
                             long M,
                             int primary_beam);

__global__ void stokes_noise_reduction(float* noise_I,
                                      int nplanes,
                                      long N,
                                      long M);

#endif  // ERRORS_KERNELS_CUH
