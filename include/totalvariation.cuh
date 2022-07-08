#ifndef TVVECTOR_CUH
#define TVVECTOR_CUH

#include "framework.cuh"
#include "functions.cuh"

class TVariation : public Fi {
 private:
  float epsilon;

 public:
  TVariation();
  TVariation(float epsilon);
  float getEpsilon();
  void setEpsilon(float epsilon);
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate(){};
};

#endif
