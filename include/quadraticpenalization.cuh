#ifndef QPVECTOR_CUH
#define QPVECTOR_CUH

#include "framework.cuh"
#include "functions.cuh"

class QuadraticP : public Fi {
 public:
  QuadraticP();
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate(){};
};

#endif
