#ifndef LAPLACIAN_CUH
#define LAPLACIAN_CUH

#include "framework.cuh"
#include "functions.cuh"

class Laplacian : public Fi {
 public:
  Laplacian();
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate(){};
};

#endif
