#ifndef VIRTUALIMAGEPROCESSOR_CUH
#define VIRTUALIMAGEPROCESSOR_CUH

#include <cuda_runtime.h>

class VirtualImageProcessor
{
public:
virtual void clipWNoise(float *I) = 0;
virtual void apply_beam(cufftComplex *image, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float freq, int primary_beam) = 0;
virtual void calculateInu(cufftComplex *image, float *I, float freq) = 0;
virtual void chainRule(float *I, float freq) = 0;
virtual void configure(int i) = 0;
protected:
float *chain;
int image_count;
};


#endif
