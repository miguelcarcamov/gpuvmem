#include <fstream>
#include <iostream>

#include "chi2.cuh"
#include "imageProcessor.cuh"

extern long M, N;
extern int image_count;
extern int flag_opt;
extern float* penalizators;
extern int nPenalizators;

Chi2::Chi2() {
  this->ip = new ImageProcessor();
  this->name = "Chi2";
};

void Chi2::configure(int penalizatorIndex,
                     int imageIndex,
                     int imageToAdd,
                     bool normalize) {
  this->imageIndex = imageIndex;
  this->order = order;
  this->mod = mod;
  this->ip->configure(image_count);
  this->normalize = normalize;

  if (penalizatorIndex != -1) {
    if (penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0) {
      printf("invalid index for penalizator (%s)\n", this->name);
      exit(-1);
    } else {
      this->penalization_factor = penalizators[penalizatorIndex];
    }
  }

  checkCudaErrors(
      cudaMalloc((void**)&result_dchi2, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMemset(result_dchi2, 0, sizeof(float) * M * N * image_count));
}

float Chi2::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(chi2(p, ip, this->normalize));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void Chi2::calcGi(float* p, float* xi) {
  dchi2(p, xi, result_dchi2, ip, this->normalize);
};

void Chi2::restartDGi() {
  checkCudaErrors(
      cudaMemset(result_dchi2, 0, sizeof(float) * M * N * image_count));
};

void Chi2::addToDphi(float* device_dphi) {
  if (image_count == 1)
    linkAddToDPhi(device_dphi, result_dchi2, 0);
  if (image_count > 1) {
    checkCudaErrors(
        cudaMemset(device_dphi, 0, sizeof(float) * M * N * image_count));
    checkCudaErrors(cudaMemcpy(device_dphi, result_dchi2,
                               sizeof(float) * N * M * image_count,
                               cudaMemcpyDeviceToDevice));
  }
};

void Chi2::setCKernel(CKernel* ckernel) {
  this->ip->setCKernel(ckernel);
};

namespace {
Fi* CreateChi2() {
  return new Chi2;
}

const std::string name = "Chi2";
const bool RegisteredChi2 =
    registerCreationFunction<Fi, std::string>(name, CreateChi2);
};  // namespace
