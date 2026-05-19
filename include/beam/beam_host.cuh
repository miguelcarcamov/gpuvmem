#ifndef BEAM_HOST_CUH
#define BEAM_HOST_CUH

// PRIMARY BEAM MODULE - Host Functions
// Handles antenna response patterns (Airy disk, Gaussian) that attenuate
// sky brightness as a function of angle from the pointing center.
// For synthesized/dirty beam (PSF) calculations, see psf/ module.

#include <cufft.h>

__host__ void linkApplyBeam2I(cufftComplex* image,
                              float antenna_diameter,
                              float pb_factor,
                              float pb_cutoff,
                              float xobs,
                              float yobs,
                              float freq,
                              int primary_beam,
                              float fg_scale);

__host__ void linkApplyBaselineBeam2I(cufftComplex* image,
                                      float ant1_diameter,
                                      float ant1_pb_factor,
                                      float ant1_pb_cutoff,
                                      int ant1_primary_beam,
                                      float ant2_diameter,
                                      float ant2_pb_factor,
                                      float ant2_pb_cutoff,
                                      int ant2_primary_beam,
                                      float xobs,
                                      float yobs,
                                      float freq,
                                      float fg_scale);

__host__ void linkClipWNoise2I(float* I);

__host__ void linkCalculateInu2I(cufftComplex* image, float* I, float freq);

__host__ void linkChain2I(float* chain, float freq, float* I, float fg_scale);

#endif  // BEAM_HOST_CUH
