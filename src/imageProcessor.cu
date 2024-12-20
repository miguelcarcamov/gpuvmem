#include "imageProcessor.cuh"

extern long N, M;

ImageProcessor::ImageProcessor() {}

void ImageProcessor::calculateInu(cufftComplex* image, float* I, float freq) {
  if (image_count == 2) {
    linkCalculateInu2I(image, I, freq);
  }
};

void ImageProcessor::apply_beam(cufftComplex* image,
                                float antenna_diameter,
                                float pb_factor,
                                float pb_cutoff,
                                float xobs,
                                float yobs,
                                float freq,
                                int primary_beam,
                                float fg_scale) {
  if (image_count == 2)
    linkApplyBeam2I(image, antenna_diameter, pb_factor, pb_cutoff, xobs, yobs,
                    freq, primary_beam, fg_scale);
};

void ImageProcessor::chainRule(float* I, float freq, float fg_scale) {
  if (image_count == 2)
    linkChain2I(chain, freq, I, fg_scale);
};

void ImageProcessor::clipWNoise(float* I) {
  if (image_count == 2)
    linkClipWNoise2I(I);
};

void ImageProcessor::configure(int i) {
  this->image_count = i;
  if (image_count > 1) {
    checkCudaErrors(
        cudaMalloc((void**)&chain, sizeof(float) * M * N * image_count));
    checkCudaErrors(cudaMemset(chain, 0, sizeof(float) * M * N * image_count));
  }
};
