#include "totalvariation.cuh"

extern long M, N;
extern int image_count;
extern float *penalizators;
extern int nPenalizators;

TVariation::TVariation() {
  this->name = "Total Variation";
  this->epsilon = 1E-12;
};

TVariation::TVariation(float epsilon) {
  this->name = "Total Variation";
  this->epsilon = epsilon;
};

float TVariation::getEpsilon() { return this->epsilon; };

void TVariation::setEpsilon(float epsilon) { this->epsilon = epsilon; };

float TVariation::calcFi(float *p) {
  float result = 0.0f;
  this->set_fivalue(totalvariation(p, device_S, this->epsilon,
                                   penalization_factor, mod, order, imageIndex,
                                   this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void TVariation::calcGi(float *p, float *xi) {
  DTVariation(p, device_DS, this->epsilon, penalization_factor, mod, order,
              imageIndex, this->iteration);
};

void TVariation::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
};

void TVariation::addToDphi(float *device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, 0);
};

void TVariation::setSandDs(float *S, float *Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi *CreateTVariation() { return new TVariation; }

const std::string name = "TotalVariation";
const bool RegisteredTVariation =
    registerCreationFunction<Fi, std::string>(name, CreateTVariation);
};  // namespace
