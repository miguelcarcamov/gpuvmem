#ifndef TVVECTOR_CUH
#define TVVECTOR_CUH

#include "framework.cuh"
#include "functions.cuh"

class IsotropicTVariation : public Fi {
 private:
  float epsilon;

 public:
  IsotropicTVariation();
  IsotropicTVariation(float epsilon);
  float getEpsilon();
  void setEpsilon(float epsilon);
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate() {};
};

class AnisotropicTVariation : public Fi {
 private:
  float epsilon;

 public:
  AnisotropicTVariation();
  AnisotropicTVariation(float epsilon);
  float getEpsilon();
  void setEpsilon(float epsilon);
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate() {};
};

#endif
