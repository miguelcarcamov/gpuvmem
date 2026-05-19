#include "objective_function/terms/regularizers/gl1norm.cuh"
#include "regularizer_kernels/regularizers_host.cuh"
#include "chi2/chi2_host.cuh"  // For linkAddToDPhi
#include "image_processing/image_processing_host.cuh"

GL1Norm::GL1Norm() {
  this->name = "G L1-Norm";
  this->prior = NULL;
  this->normalization_factor = 1.0f;
  this->epsilon_a = 1E-12;
  this->epsilon_b = 1E-12;
};

GL1Norm::GL1Norm(std::vector<float> prior) {
  this->name = "G L1-Norm";
  this->normalization_factor = 1.0f;
  this->epsilon_a = 1E-12;
  this->epsilon_b = 1E-12;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
};

GL1Norm::GL1Norm(std::vector<float> prior, float epsilon_a, float epsilon_b) {
  this->name = "G L1-Norm";
  this->normalization_factor = 1.0f;
  this->epsilon_a = epsilon_a;
  this->epsilon_b = epsilon_b;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
};

GL1Norm::GL1Norm(float* prior, float normalization_factor) {
  this->name = "G L1-Norm";
  this->prior = prior;
  this->epsilon_a = 1E-12;
  this->epsilon_b = 1E-12;
  this->normalization_factor = normalization_factor;
  this->normalizePrior();
};

GL1Norm::GL1Norm(float* prior,
                 float normalization_factor,
                 float epsilon_a,
                 float epsilon_b) {
  this->name = "G L1-Norm";
  this->prior = prior;
  this->epsilon_a = epsilon_a;
  this->epsilon_b = epsilon_b;
  this->normalization_factor = normalization_factor;
  this->normalizePrior();
};

GL1Norm::GL1Norm(std::vector<float> prior, float normalization_factor) {
  this->name = "G L1-Norm";
  this->epsilon_a = 1E-12;
  this->epsilon_b = 1E-12;
  this->normalization_factor = normalization_factor;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  this->normalizePrior();
};

GL1Norm::GL1Norm(std::vector<float> prior,
                 float normalization_factor,
                 float epsilon_a,
                 float epsilon_b) {
  this->name = "G L1-Norm";
  this->epsilon_a = epsilon_a;
  this->epsilon_b = epsilon_b;
  this->normalization_factor = normalization_factor;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  this->normalizePrior();
};

GL1Norm::GL1Norm(float* prior) {
  this->name = "G L1-Norm";
  this->prior = prior;
  this->normalization_factor = 1.0f;
  this->epsilon_a = 1E-12;
  this->epsilon_b = 1E-12;
};

GL1Norm::GL1Norm(float* prior, float epsilon_a, float epsilon_b) {
  this->name = "G L1-Norm";
  this->prior = prior;
  this->normalization_factor = 1.0f;
  this->epsilon_a = epsilon_a;
  this->epsilon_b = epsilon_b;
};

GL1Norm::~GL1Norm() {
  cudaFree(this->prior);
};

float GL1Norm::getNormalizationFactor() {
  return this->normalization_factor;
};

void GL1Norm::setNormalizationFactor(float normalization_factor) {
  this->normalization_factor = normalization_factor;
};

void GL1Norm::setPrior(float* prior) {
  cudaFree(this->prior);
  this->prior = prior;
};

void GL1Norm::setEpsilonA(float epsilon) {
  this->epsilon_a = epsilon;
};

void GL1Norm::setEpsilonB(float epsilon) {
  this->epsilon_b = epsilon;
};

void GL1Norm::setEpsilons(float epsilon_a, float epsilon_b) {
  this->epsilon_a = epsilon_a;
  this->epsilon_b = epsilon_b;
};

float GL1Norm::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(GL1NormK(p, this->prior, device_S, penalization_factor,
                             this->epsilon_a, this->epsilon_b, mod, order,
                             imageIndex, this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void GL1Norm::calcGi(float* p, float* xi) {
  DGL1Norm(p, device_DS, this->prior, penalization_factor, this->epsilon_a,
           this->epsilon_b, mod, order, imageIndex, this->iteration);
};

void GL1Norm::restartDGi() {
  const size_t plane = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN());
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * plane));
};

void GL1Norm::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void GL1Norm::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

void GL1Norm::normalizePrior() {
  normalizeImage(this->prior, this->normalization_factor);
};

namespace {
Fi* CreateGL1Norm() {
  return new GL1Norm;
}
const std::string name = "GL1Norm";
const bool RegisteredGL1Norm =
    registerCreationFunction<Fi, std::string>(name, CreateGL1Norm);
const bool RegisteredGL1NormInt =
    registerCreationFunction<Fi, int>(0, CreateGL1Norm);
};  // namespace
