#include "objective_function/terms/regularizers/totalsquaredvariation.cuh"
#include "regularizer_kernels/regularizers_host.cuh"
#include "chi2/chi2_host.cuh"  // For linkAddToDPhi

TotalSquaredVariationP::TotalSquaredVariationP() {
  this->name = "Total Squared Variation";
};

float TotalSquaredVariationP::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(TotalSquaredVariation(p, device_S, penalization_factor, mod,
                                          order, imageIndex, this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
}
void TotalSquaredVariationP::calcGi(float* p, float* xi) {
  DTSVariation(p, device_DS, penalization_factor, mod, order, imageIndex,
               this->iteration);
};

void TotalSquaredVariationP::restartDGi() {
  const size_t plane = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN());
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * plane));
};

void TotalSquaredVariationP::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void TotalSquaredVariationP::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi* CreateTotalSquaredVariation() {
  return new TotalSquaredVariationP;
}

const std::string name = "TotalSquaredVariation";
const bool RegisteredTotalSquaredVariation =
    registerCreationFunction<Fi, std::string>(name,
                                              CreateTotalSquaredVariation);
};  // namespace
