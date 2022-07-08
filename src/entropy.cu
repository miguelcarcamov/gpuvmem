#include "entropy.cuh"

extern long M, N;
extern int image_count;
extern float *penalizators;
extern int nPenalizators;

Entropy::Entropy() {
  this->name = "Entropy";
  this->prior_value = 1.0f;
  this->eta = -1.0f;
};

Entropy::Entropy(float prior_value) {
  this->name = "Entropy";
  this->prior_value = prior_value;
  this->eta = -1.0f;
};

Entropy::Entropy(float prior_value, float eta) {
  this->name = "Entropy";
  this->prior_value = prior_value;
  this->eta = eta;
};

float Entropy::getPrior() { return this->prior_value; };

void Entropy::setPrior(float prior_value) { this->prior_value = prior_value; };

float Entropy::getEta() { return this->eta; };

void Entropy::setEta(float eta) { this->eta = eta; };

float Entropy::calcFi(float *p) {
  float result = 0.0f;
  this->set_fivalue(SEntropy(p, device_S, this->prior_value, this->eta,
                             penalization_factor, mod, order, imageIndex,
                             this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void Entropy::calcGi(float *p, float *xi) {
  DEntropy(p, device_DS, this->prior_value, this->eta, penalization_factor, mod,
           order, imageIndex, this->iteration);
};

void Entropy::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
};

void Entropy::addToDphi(float *device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void Entropy::setSandDs(float *S, float *Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi *CreateEntropy() { return new Entropy; }
const std::string name = "Entropy";
const bool RegisteredEntropy =
    registerCreationFunction<Fi, std::string>(name, CreateEntropy);
const bool RegisteredEntropyInt =
    registerCreationFunction<Fi, int>(0, CreateEntropy);
};  // namespace
