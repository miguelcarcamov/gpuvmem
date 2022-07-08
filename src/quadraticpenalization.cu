#include "quadraticpenalization.cuh"

extern long M, N;
extern int image_count;
extern float* penalizators;
extern int nPenalizators;

QuadraticP::QuadraticP(){};

float QuadraticP::calcFi(float* p) {
  float result = 0.0;
  this->set_fivalue(quadraticP(p, device_S, penalization_factor, mod, order,
                               imageIndex, this->iteration));
  result = (penalization_factor) * (this->get_fivalue());
  return result;
}
void QuadraticP::calcGi(float* p, float* xi) {
  DQuadraticP(p, device_DS, penalization_factor, mod, order, imageIndex,
              this->iteration);
};

void QuadraticP::restartDGi() {
  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
};

void QuadraticP::addToDphi(float* device_dphi) {
  linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void QuadraticP::setSandDs(float* S, float* Ds) {
  cudaFree(this->device_S);
  cudaFree(this->device_DS);
  this->device_S = S;
  this->device_DS = Ds;
};

namespace {
Fi* CreateQuadraticP() {
  return new QuadraticP;
}

const std::string name = "Quadratic";
const bool RegisteredQuadraticP =
    registerCreationFunction<Fi, std::string>(name, CreateQuadraticP);
};  // namespace
