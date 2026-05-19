#include "objective_function/terms/regularizers/gentropy.cuh"
#include "regularizer_kernels/regularizers_host.cuh"
#include "chi2/chi2_host.cuh"  // For linkAddToDPhi
#include "image_processing/image_processing_host.cuh"

GEntropy::GEntropy() {
  this->name = "GEntropy";
  this->prior = NULL;
  this->normalization_factor = 1.0f;
  this->eta = -1.0f;
};

GEntropy::GEntropy(std::vector<float> prior) {
  this->name = "GEntropy";
  this->normalization_factor = 1.0f;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  this->eta = -1.0f;
};

GEntropy::GEntropy(float* prior, float normalization_factor) {
  this->name = "GEntropy";
  this->prior = prior;
  this->normalization_factor = normalization_factor;
  this->normalizePrior();
  this->eta = -1.0f;
};

GEntropy::GEntropy(float* prior, float normalization_factor, float eta) {
  this->name = "GEntropy";
  this->prior = prior;
  this->normalization_factor = normalization_factor;
  this->normalizePrior();
  this->eta = eta;
};

GEntropy::GEntropy(std::vector<float> prior, float normalization_factor) {
  this->name = "GEntropy";
  this->normalization_factor = normalization_factor;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  this->normalizePrior();
  this->eta = -1.0f;
};

GEntropy::GEntropy(std::vector<float> prior,
                   float normalization_factor,
                   float eta) {
  this->name = "GEntropy";
  this->normalization_factor = normalization_factor;
  const size_t n = prior.size();
  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * n));
  checkCudaErrors(
      cudaMemcpy(this->prior, prior.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  this->normalizePrior();
  this->eta = eta;
};

GEntropy::GEntropy(float* prior) {
  this->name = "GEntropy";
  this->prior = prior;
  this->normalization_factor = 1.0f;
  this->eta = -1.0f;
};

GEntropy::~GEntropy() {
  cudaFree(this->prior);
};

float GEntropy::getNormalizationFactor() {
  return this->normalization_factor;
};

void GEntropy::setNormalizationFactor(float normalization_factor) {
  this->normalization_factor = normalization_factor;
};

float GEntropy::getEta() {
  return this->eta;
};

void GEntropy::setEta(float eta) {
  this->eta = eta;
};

void GEntropy::setPrior(float* prior) {
  cudaFree(this->prior);
  this->prior = prior;
};

float GEntropy::calcFi(float* p) {
  float result = 0.0f;
  this->set_fivalue(SGEntropy(p, device_S, this->prior, this->eta,
                              penalization_factor, mod, order, imageIndex,
                              this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
};

void GEntropy::calcGi(float* p, float* xi) {
  DGEntropy(p, device_DS, this->prior, this->eta, penalization_factor, mod,
            order, imageIndex, this->iteration);
};

void GEntropy::restartDGi() {
  const size_t plane = static_cast<size_t>(gridM()) * static_cast<size_t>(gridN());
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * plane));
};

void GEntropy::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void GEntropy::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

void GEntropy::normalizePrior() {
  normalizeImage(this->prior, this->normalization_factor);
};

namespace {
Fi* CreateGEntropy() {
  return new GEntropy;
}
const std::string name = "GEntropy";
const bool RegisteredGEntropy =
    registerCreationFunction<Fi, std::string>(name, CreateGEntropy);
const bool RegisteredGEntropyInt =
    registerCreationFunction<Fi, int>(0, CreateGEntropy);
};  // namespace
