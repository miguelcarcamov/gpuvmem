#ifndef VIRTUALIMAGEPROCESSOR_CUH
#define VIRTUALIMAGEPROCESSOR_CUH

#include <cuda_runtime.h>

class VirtualImageProcessor
{
public:
    virtual void clip(float *I) = 0;
    virtual void clipWNoise(float *I) = 0;
    virtual void apply_beam(cufftComplex *image, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float freq, int primary_beam) = 0;
    virtual void calculateInu(cufftComplex *image, float *I, float freq) = 0;
    virtual void chainRule(float *I, float freq) = 0;
    virtual void configure(int I) = 0;
protected:
    float *chain;
    cufftComplex *fg_image;
    int image_count;
};


#endif
