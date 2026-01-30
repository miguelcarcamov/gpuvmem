#ifndef VIRTUALIMAGEPROCESSOR_CUH
#define VIRTUALIMAGEPROCESSOR_CUH

#include <cuda_runtime.h>

#include "ckernel.cuh"

class Image;

class VirtualImageProcessor {
 public:
  virtual void clipWNoise(float* I) = 0;
  virtual void apply_beam(cufftComplex* image,
                          float antenna_diameter,
                          float pb_factor,
                          float pb_cutoff,
                          float xobs,
                          float yobs,
                          float freq,
                          int primary_beam,
                          float fg_scale) = 0;
  virtual void calculateInu(cufftComplex* image, float* I, float freq) = 0;
  virtual void chainRule(float* I, float freq, float fg_scale) = 0;
  /** Configure from Image geometry (dimensions and image count). Call when image is created (e.g. from setDevice). */
  virtual void configure(Image* img) { (void)img; }
  virtual CKernel* getCKernel() { return this->ckernel; };
  virtual void setCKernel(CKernel* ckernel) { this->ckernel = ckernel; };

 protected:
  float* chain;
  int image_count;
  CKernel* ckernel = NULL;
};

#endif
