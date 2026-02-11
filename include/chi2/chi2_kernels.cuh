#ifndef CHI2_KERNELS_CUH
#define CHI2_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>

struct double3;

__global__ void chi2Vector(float* __restrict__ chi2,
                           const cufftComplex* __restrict__ Vr,
                           const float* __restrict__ w,
                           long numVisibilities);

__global__ void weightsSquaredVector(float* __restrict__ w_squared,
                                     const float* __restrict__ w,
                                     long numVisibilities);

__global__ void DChi2(float* noise,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize,
                      float N_eff);

__global__ void DChi2(float* noise,
                      float* gcf,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize,
                      float N_eff);

// Gather one (chan, pol) visibility from each chunk at offset into contiguous buffers
__global__ void gatherChunkAtOffset(double3* uvw_out,
                                    cufftComplex* Vo_out,
                                    float* weight_out,
                                    double3 const* const* uvw_ptrs,
                                    cufftComplex const* const* Vo_ptrs,
                                    float const* const* weight_ptrs,
                                    int offset,
                                    int n);

// Add gradient contribution to multi-plane gradient array
__global__ void AddToDPhi(float* dphi, float* dgi, long N, long M, int index);

#endif  // CHI2_KERNELS_CUH
