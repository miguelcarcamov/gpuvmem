#include "l2constantprior.cuh"

extern long M, N;
extern int image_count;
extern float* penalizators;
extern int nPenalizators;

L2ConstantPrior::L2ConstantPrior() {
  this->name = "L2ConstantPrior";
  this->prior_value = 0.0f;
}

L2ConstantPrior::L2ConstantPrior(float prior_value) {
  this->name = "L2ConstantPrior";
  this->prior_value = prior_value;
}

float L2ConstantPrior::getPrior() {
  return this->prior_value;
}

void L2ConstantPrior::setPrior(float prior_value) {
  this->prior_value = prior_value;
}

float L2ConstantPrior::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(l2ConstantPrior(p, device_S, this->prior_value,
                                    penalization_factor, mod, order, imageIndex,
                                    this->iteration));
  result = penalization_factor * this->get_fivalue();
  return result;
}

void L2ConstantPrior::calcGi(float* p, float* xi) {
  DL2ConstantPrior(p, device_DS, this->prior_value, penalization_factor, mod,
                   order, imageIndex, this->iteration);
}

void L2ConstantPrior::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
}

void L2ConstantPrior::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
}

void L2ConstantPrior::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
}

namespace {
Fi* CreateL2ConstantPrior() {
  return new L2ConstantPrior;
}
const std::string name = "L2ConstantPrior";
const bool RegisteredL2ConstantPrior =
    registerCreationFunction<Fi, std::string>(name, CreateL2ConstantPrior);
};  // namespace
