#include "imageProcessor.cuh"

extern long N, M;

ImageProcessor::ImageProcessor() {
  this->fg_scale = 0.0f;
  this->spec_index_noise = 1.0f;
}

ImageProcessor::ImageProcessor(float fg_scale, float spec_index_noise) {
  this->fg_scale = fg_scale;
  this->spec_index_noise = spec_index_noise;
}

float ImageProcessor::getFgScale() {
  return this->fg_scale;
}

void ImageProcessor::setFgScale(float fg_scale) {
  this->fg_scale = fg_scale;
}

float ImageProcessor::getSpectralIndexNoise() {
  return this->spec_index_noise;
}

void ImageProcessor::setSpectralIndexNoise(float spec_index_noise) {
  this->spec_index_noise = spec_index_noise;
}

void ImageProcessor::calculateInu(cufftComplex* image, float* I, float freq) {
  if (image_count == 2) {
    linkCalculateInu2I(image, I, freq, this->spec_index_noise);
  }
};

void ImageProcessor::apply_beam(cufftComplex* image,
                                float antenna_diameter,
                                float pb_factor,
                                float pb_cutoff,
                                float xobs,
                                float yobs,
                                float freq,
                                int primary_beam) {
  if (image_count == 2)
    linkApplyBeam2I(image, antenna_diameter, pb_factor, pb_cutoff, xobs, yobs,
                    freq, primary_beam);
};

void ImageProcessor::chainRule(float* I, float freq) {
  if (image_count == 2)
    linkChain2I(chain, freq, I);
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
