#include "imageProcessor.cuh"
#include "classes/image.cuh"

ImageProcessor::ImageProcessor() {}

void ImageProcessor::calculateInu(cufftComplex* image, float* I, float freq) {
  if (image_count == 2) {
    linkCalculateInu2I(image, I, freq);
  } else {
    // Stokes imaging (image_count == npol) or single image: I is one plane
    linkCopyItoInu(image, I);
  }
}

void ImageProcessor::apply_beam(cufftComplex* image,
                                float antenna_diameter,
                                float pb_factor,
                                float pb_cutoff,
                                float xobs,
                                float yobs,
                                float freq,
                                int primary_beam,
                                float fg_scale) {
  // Apply beam for both MFS (image_count == 2) and Stokes/single-image
  linkApplyBeam2I(image, antenna_diameter, pb_factor, pb_cutoff, xobs, yobs,
                  freq, primary_beam, fg_scale);
}

void ImageProcessor::chainRule(float* I, float freq, float fg_scale) {
  if (image_count == 2)
    linkChain2I(chain, freq, I, fg_scale);
};

void ImageProcessor::clipWNoise(float* I) {
  if (image_count == 2)
    linkClipWNoise2I(I);
  else
    linkClipStokesWNoise(I, image_count);
}

void ImageProcessor::configure(Image* img) {
  if (!img) return;
  this->image_count = img->getImageCount();
  long M_img = img->getM();
  long N_img = img->getN();
  if (image_count > 1 && M_img > 0 && N_img > 0) {
    checkCudaErrors(cudaMalloc((void**)&chain,
                               sizeof(float) * M_img * N_img * image_count));
    checkCudaErrors(cudaMemset(chain, 0,
                              sizeof(float) * M_img * N_img * image_count));
  }
}
