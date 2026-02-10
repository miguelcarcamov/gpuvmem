#ifndef CHAIN_RULE_CUH
#define CHAIN_RULE_CUH

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void DChi2_total_alpha(float* noise,
                                  float* dchi2_total,
                                  float* dchi2,
                                  float* I,
                                  float nu,
                                  float nu_0,
                                  float noise_cut,
                                  long N,
                                  long M);

__global__ void DChi2_total_I_nu_0(float* noise,
                                   float* dchi2_total,
                                   float* dchi2,
                                   float* I,
                                   float nu,
                                   float nu_0,
                                   float noise_cut,
                                   long N,
                                   long M);

__global__ void chainRule2ISimplified(float* chain,
                                      float* noise,
                                      float* I,
                                      float nu,
                                      float nu_0,
                                      float noise_cut,
                                      float fg_scale,
                                      long N,
                                      long M);

__global__ void chainRule2I(float* chain,
                            float* noise,
                            float* I,
                            float nu,
                            float nu_0,
                            float noise_cut,
                            float fg_scale,
                            float ref_xobs,
                            float ref_yobs,
                            double DELTAX,
                            double DELTAY,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            int primary_beam,
                            long N,
                            long M);

__global__ void DChi2_2I(float* noise,
                         float* chain,
                         float* I,
                         float* dchi2,
                         float* dchi2_total,
                         float threshold,
                         float alpha_n_sigma,
                         float noise_cut,
                         int image,
                         long N,
                         long M);

#endif  // CHAIN_RULE_CUH
