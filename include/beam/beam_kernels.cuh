#ifndef BEAM_KERNELS_CUH
#define BEAM_KERNELS_CUH

// PRIMARY BEAM MODULE
// Handles antenna response patterns (Airy disk, Gaussian) that attenuate
// sky brightness as a function of angle from the pointing center.
// This is distinct from the synthesized/dirty beam (PSF) which characterizes
// image resolution from uv-coverage (see psf/ module).

#include <cufft.h>
#include <cuda_runtime.h>

// Device beam functions
__device__ float AiryDiskBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor);

__device__ float GaussianBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor);

__device__ float attenuation(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float freq,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam);

__device__ cufftComplex WKernel(double w, float xobs, float yobs, double DELTAX, double DELTAY);

// Beam kernels
__global__ void distance_image(float* distance_image,
                               float xobs,
                               float yobs,
                               float dist_arcsec,
                               double DELTAX,
                               double DELTAY,
                               long N);

__global__ void total_attenuation(float* total_atten,
                                  float antenna_diameter,
                                  float pb_factor,
                                  float pb_cutoff,
                                  float freq,
                                  float xobs,
                                  float yobs,
                                  double DELTAX,
                                  double DELTAY,
                                  long N,
                                  int primary_beam);

__global__ void weight_image(float* weight_image, float* total_atten, long N);

__global__ void noise_image(float* noise_image,
                            float* weight_image,
                            float max_weight,
                            float noise_jypix,
                            long N);

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float fg_scale,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam);

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float* gcf,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam);

__global__ void apply_GCF(cufftComplex* __restrict__ image,
                          const float* __restrict__ gcf,
                          long N);

// Spectral model kernel: Calculate I_ν from I_ν₀ and spectral index α
__global__ void calculateInu(cufftComplex* I_nu,
                             float* I,
                             float nu,
                             float nu_0,
                             float MINPIX,
                             float eta,
                             long N,
                             long M);

#endif  // BEAM_KERNELS_CUH
