#include "objective_function/terms/regularizers/laplacian.cuh"
#include "regularizer_kernels/regularizers_host.cuh"
#include "chi2/chi2_host.cuh"  // For linkAddToDPhi

Laplacian::Laplacian() {
  this->name = "Laplacian";
};

float Laplacian::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(laplacian(p, device_S, penalization_factor, mod, order,
                              imageIndex, this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
}
void Laplacian::calcGi(float* p, float* xi) {
  DLaplacian(p, device_DS, penalization_factor, mod, order, imageIndex,
             this->iteration);
};

void Laplacian::restartDGi() {
  const size_t plane = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN());
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * plane));
};

void Laplacian::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void Laplacian::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi* CreateLaplacian() {
  return new Laplacian;
}
const std::string name = "Laplacian";
const bool RegisteredLaplacian =
    registerCreationFunction<Fi, std::string>(name, CreateLaplacian);
};  // namespace
