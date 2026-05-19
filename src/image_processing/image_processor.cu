#include "image_processing/imageProcessor.cuh"
#include "beam/beam_host.cuh"
#include "framework.cuh"

ImageProcessor::ImageProcessor() {
  image_count = 0;
  chain = nullptr;
}

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

void ImageProcessor::apply_baseline_beam(cufftComplex* image,
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
                                           float fg_scale) {
  if (image_count == 2)
    linkApplyBaselineBeam2I(image, ant1_diameter, ant1_pb_factor, ant1_pb_cutoff,
                          ant1_primary_beam, ant2_diameter, ant2_pb_factor,
                          ant2_pb_cutoff, ant2_primary_beam, xobs, yobs, freq,
                          fg_scale);
};

void ImageProcessor::chainRule(float* I, float freq, float fg_scale) {
  if (image_count == 2)
    linkChain2I(chain, freq, I, fg_scale);
};

void ImageProcessor::clipWNoise(float* I) {
  if (image_count == 2)
    linkClipWNoise2I(I);
};

void ImageProcessor::configure(Image* img) {
  if (!img) return;
  
  this->image_count = img->getImageCount();
  long M = img->getM();
  long N = img->getN();
  
  if (image_count > 1) {
    // Free existing chain if reconfiguring
    if (chain) {
      checkCudaErrors(cudaFree(chain));
      chain = nullptr;
    }
    checkCudaErrors(
        cudaMalloc((void**)&chain, sizeof(float) * M * N * image_count));
    checkCudaErrors(cudaMemset(chain, 0, sizeof(float) * M * N * image_count));
  }
};
