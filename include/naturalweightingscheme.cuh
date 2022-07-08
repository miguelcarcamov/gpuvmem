#ifndef NATURALWEIGHTINGSCHEME_CUH
#define NATURALWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

class NaturalWeightingScheme : public WeightingScheme {
 public:
  NaturalWeightingScheme();
  NaturalWeightingScheme(int threads);
  NaturalWeightingScheme(int threads, UVTaper* uvtaper);
  void configure(void* params){};
  void apply(std::vector<MSDataset>& d);
};

#endif
