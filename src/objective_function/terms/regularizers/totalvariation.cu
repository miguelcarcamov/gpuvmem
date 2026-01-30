#include "objective_function/terms/regularizers/totalvariation.cuh"

extern long M, N;
extern int image_count;
extern float* penalizators;
extern int nPenalizators;

IsotropicTVariation::IsotropicTVariation() {
  this->name = "Isotropic Total Variation";
  this->epsilon = 1E-12;
};

IsotropicTVariation::IsotropicTVariation(float epsilon) {
  this->name = "Isotropic Total Variation";
  this->epsilon = epsilon;
};

float IsotropicTVariation::getEpsilon() {
  return this->epsilon;
};

void IsotropicTVariation::setEpsilon(float epsilon) {
  this->epsilon = epsilon;
};

float IsotropicTVariation::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(isotropicTV(p, device_S, this->epsilon, penalization_factor,
                                mod, order, imageIndex, this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void IsotropicTVariation::calcGi(float* p, float* xi) {
  DIsotropicTV(p, device_DS, this->epsilon, penalization_factor, mod, order,
               imageIndex, this->iteration);
};

void IsotropicTVariation::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
};

void IsotropicTVariation::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, 0);
};

void IsotropicTVariation::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

AnisotropicTVariation::AnisotropicTVariation() {
  this->name = "Anisotropic Total Variation";
  this->epsilon = 1E-12;
};

AnisotropicTVariation::AnisotropicTVariation(float epsilon) {
  this->name = "Anisotropic Total Variation";
  this->epsilon = epsilon;
};

float AnisotropicTVariation::getEpsilon() {
  return this->epsilon;
};

void AnisotropicTVariation::setEpsilon(float epsilon) {
  this->epsilon = epsilon;
};

float AnisotropicTVariation::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(anisotropicTV(p, device_S, this->epsilon,
                                  penalization_factor, mod, order, imageIndex,
                                  this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void AnisotropicTVariation::calcGi(float* p, float* xi) {
  DAnisotropicTV(p, device_DS, this->epsilon, penalization_factor, mod, order,
                 imageIndex, this->iteration);
};

void AnisotropicTVariation::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
};

void AnisotropicTVariation::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, 0);
};

void AnisotropicTVariation::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi* CreateIsotropicTVariation() {
  return new IsotropicTVariation;
}

const std::string name_isotropic = "IsotropicTotalVariation";
const bool RegisteredIsotropicTVariation =
    registerCreationFunction<Fi, std::string>(name_isotropic,
                                              CreateIsotropicTVariation);

Fi* CreateAnisotropicTVariation() {
  return new AnisotropicTVariation;
}

const std::string name_anisotropic = "AnisotropicTotalVariation";
const bool RegisteredAnisotropicTVariation =
    registerCreationFunction<Fi, std::string>(name_anisotropic,
                                              CreateAnisotropicTVariation);
};  // namespace
