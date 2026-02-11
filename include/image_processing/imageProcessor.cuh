#ifndef IMAGE_PROCESSOR_CUH
#define IMAGE_PROCESSOR_CUH

#include "framework.cuh"
#include "classes/image.cuh"

class ImageProcessor : public VirtualImageProcessor {
 public:
  ImageProcessor();
  void clipWNoise(float* I);
  void apply_beam(cufftComplex* image,
                  float antenna_diameter,
                  float pb_factor,
                  float pb_cutoff,
                  float xobs,
                  float yobs,
                  float freq,
                  int primary_beam,
                  float fg_scale);
  void calculateInu(cufftComplex* image, float* I, float freq);
  void chainRule(float* I, float freq, float fg_scale);
  /** Configure from Image geometry (replaces legacy configure(int)). */
  void configure(Image* img) override;
};

#endif
