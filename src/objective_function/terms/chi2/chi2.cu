#include <fstream>
#include <iostream>

#include "objective_function/terms/chi2/chi2.cuh"
#include <iostream>
#include "chi2/chi2_host.cuh"  // For linkAddToDPhi
#include "classes/image.cuh"
#include "image_processing/imageProcessor.cuh"

Chi2::Chi2() {
  this->ip = new ImageProcessor();
  this->name = "Chi2";
  this->normalize = false;
};

void Chi2::configure(int penalizatorIndex,
                     int imageIndex,
                     int imageToAdd,
                     bool normalize) {
  this->normalize = normalize;
  /* ImageProcessor is configured from Image in configureImage() when image is created (e.g. from setDevice). */

  if (penalizatorIndex != -1) {
    if (penalizatorIndex > (zWeightCount() - 1) || penalizatorIndex < 0) {
      std::cerr << "Chi2: invalid image index for term \"" << this->name << "\"\n";
      exit(-1);
    } else if (zWeights() != nullptr) {
      this->penalization_factor = zWeights()[penalizatorIndex];
    } else {
      this->penalization_factor = 0.0f;
    }
  }

  Fi::configure(-1, imageIndex, imageToAdd, normalize);

  const size_t vol = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN()) *
                     static_cast<size_t>(gridImageCount());
  checkCudaErrors(cudaMalloc((void**)&result_dchi2, sizeof(float) * vol));
  checkCudaErrors(cudaMemset(result_dchi2, 0, sizeof(float) * vol));
}

void Chi2::configureImage(Image* image) {
  image_ = image;
  if (image && ip)
    ip->configure(image);
}

float Chi2::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(
      chi2(p, image_, ip, this->normalize, this->fg_scale));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void Chi2::calcGi(float* p, float* xi) {
  dchi2(p, xi, result_dchi2, image_, ip, this->normalize, this->fg_scale);
};

void Chi2::restartDGi() {
  const size_t vol = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN()) *
                     static_cast<size_t>(gridImageCount());
  checkCudaErrors(cudaMemset(result_dchi2, 0, sizeof(float) * vol));
};

void Chi2::addToDphi(float* device_dphi) {
  const int nic = gridImageCount();
  if (nic == 1)
    linkAddToDPhi(device_dphi, result_dchi2, 0);
  if (nic > 1) {
    const size_t vol = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN()) *
                       static_cast<size_t>(nic);
    checkCudaErrors(cudaMemset(device_dphi, 0, sizeof(float) * vol));
    checkCudaErrors(cudaMemcpy(device_dphi, result_dchi2, sizeof(float) * vol,
                               cudaMemcpyDeviceToDevice));
  }
};

void Chi2::setCKernel(CKernel* ckernel) {
  this->ip->setCKernel(ckernel);
};

void Chi2::setFgScale(float fg_scale) {
  this->fg_scale = fg_scale;
};

float Chi2::getFgScale() {
  return this->fg_scale;
};

namespace {
Fi* CreateChi2() {
  return new Chi2;
}

const std::string name = "Chi2";
const bool RegisteredChi2 =
    registerCreationFunction<Fi, std::string>(name, CreateChi2);
};  // namespace
