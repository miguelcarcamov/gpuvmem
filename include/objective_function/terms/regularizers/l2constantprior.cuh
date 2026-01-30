#ifndef L2CONSTANTPRIOR_CUH
#define L2CONSTANTPRIOR_CUH

#include "framework.cuh"
#include "functions.cuh"

class L2ConstantPrior : public Fi {
 private:
  float prior_value;

 public:
  L2ConstantPrior();
  explicit L2ConstantPrior(float prior_value);
  float getPrior();
  void setPrior(float prior_value) override;
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void setSandDs(float* S, float* Ds);
  float calculateSecondDerivate() { return 0.0f; }
};

#endif
