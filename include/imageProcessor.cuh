#ifndef IMAGE_PROCESSOR_CUH
#define IMAGE_PROCESSOR_CUH

#include "framework.cuh"
#include "functions.cuh"

class ImageProcessor : public VirtualImageProcessor {
 public:
  ImageProcessor();
  ImageProcessor(float fg_scale, float spec_index_noise);
  void clipWNoise(float* I);
  void apply_beam(cufftComplex* image,
                  float antenna_diameter,
                  float pb_factor,
                  float pb_cutoff,
                  float xobs,
                  float yobs,
                  float freq,
                  int primary_beam);
  void calculateInu(cufftComplex* image, float* I, float freq);
  void chainRule(float* I, float freq);
  void configure(int i);
  float getFgScale();
  void setFgScale(float fg_scale);
  float getSpectralIndexNoise();
  void setSpectralIndexNoise(float spec_index_noise);

 private:
  float fg_scale;
  float spec_index_noise;
};

#endif
