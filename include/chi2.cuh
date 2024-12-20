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
  void configure(int penalizatorIndex,
                 int imageIndex,
                 int imageToAdd,
                 bool normalize) override;
  void setPenalizationFactorFromInputIndex(int index) {};
  float calculateSecondDerivate() {};
  void setCKernel(CKernel* ckernel) override;
  void setFgScale(float fg_scale) override;
  float getFgScale() override;

 private:
  VirtualImageProcessor* ip;
  float fg_scale = 1.0;
  int imageToAdd;
  float* result_dchi2;
};

#endif
