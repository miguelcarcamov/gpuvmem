#ifndef CHI2_CUH
#define CHI2_CUH

#include "framework.cuh"
#include "functions.cuh"

class Chi2 : public Fi {
 public:
  Chi2();
  float calcFi(float* p);
  void calcGi(float* p, float* xi);
  void restartDGi();
  void addToDphi(float* device_dphi);
  void configure(int penalizatorIndex, int imageIndex, int imageToAdd) override;
  void setPenalizationFactorFromInputIndex(int index){};
  float calculateSecondDerivate(){};
  void setCKernel(CKernel* ckernel) override;
  VirtualImageProcessor* getVirtualImageProcessor();

 private:
  VirtualImageProcessor* ip;
  int imageToAdd;
  float* result_dchi2;
};

#endif
