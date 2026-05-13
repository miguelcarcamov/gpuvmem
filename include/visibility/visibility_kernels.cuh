#ifndef VISIBILITY_KERNELS_CUH
#define VISIBILITY_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>

struct double3;

// Device functions for phase rotation
__device__ void computeFrequenciesAndPhaseCenter(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double reference_column,
                                                 double reference_row,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative);

__device__ void computeFrequenciesAndPhaseCorner(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative);

// Device functions for bilinear interpolation
__device__ bool interpolateVisibilityCenter(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result);

__device__ bool interpolateVisibilityCorner(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result);

// Visibility kernels
__global__ void phase_rotate(cufftComplex* __restrict__ data,
                             long M,
                             long N,
                             double xphs,
                             double yphs,
                             double reference_column,
                             double reference_row,
                             bool dc_at_center);

__global__ void bilinearInterpolateVisibility(
    cufftComplex* __restrict__ Vm,
    const cufftComplex* __restrict__ V,
    const double3* __restrict__ UVW,
    float* __restrict__ weight,
    const double deltau,
    const double deltav,
    const long numVisibilities,
    const long M,
    const long N,
    const bool dc_at_center);

__global__ void residual(cufftComplex* __restrict__ Vr,
                         const cufftComplex* __restrict__ Vm,
                         const cufftComplex* __restrict__ Vo,
                         long numVisibilities);

#endif  // VISIBILITY_KERNELS_CUH
