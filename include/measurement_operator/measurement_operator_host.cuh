#ifndef MEASUREMENT_OPERATOR_HOST_CUH
#define MEASUREMENT_OPERATOR_HOST_CUH

#include "image_processing/imageProcessor.cuh"
#include "classes/ckernel.cuh"

// Forward declarations
struct varsPerGPU;

// Measurement operator: transforms image to visibility grid
// This encapsulates the forward model pipeline:
//   calculateInu -> apply_beam -> apply_GCF -> FFT2D -> phase_rotate
__host__ void computeImageToVisibilityGrid(float* I,
                                           VirtualImageProcessor* ip,
                                           varsPerGPU* vars_gpu,
                                           int gpu_idx,
                                           long M,
                                           long N,
                                           float nu,
                                           float ref_xobs_pix,
                                           float ref_yobs_pix,
                                           float phs_xobs_pix,
                                           float phs_yobs_pix,
                                           float antenna_diameter,
                                           float pb_factor,
                                           float pb_cutoff,
                                           int primary_beam,
                                           float fg_scale,
                                           CKernel* ckernel,
                                           bool fft_shift);

#endif  // MEASUREMENT_OPERATOR_HOST_CUH
