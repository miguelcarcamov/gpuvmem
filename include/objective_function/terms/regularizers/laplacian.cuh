#ifndef LAPLACIAN_CUH
#define LAPLACIAN_CUH

#include "framework.cuh"
#include "framework.cuh"

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
