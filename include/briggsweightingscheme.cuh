#ifndef BRIGGSWEIGHTINGSCHEME_CUH
#define BRIGGSWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

extern long M;
extern long N;
extern double deltau;
extern double deltav;

class BriggsWeightingScheme : public WeightingScheme {
 public:
  BriggsWeightingScheme();
  BriggsWeightingScheme(int threads);
  BriggsWeightingScheme(int threads, UVTaper* uvtaper);
  void apply(std::vector<MSDataset>& d);
  void configure(void* params);
  float getRobustParam();
  void setRobustParam(float robust_param);

 private:
  float robust_param;
};

#endif
