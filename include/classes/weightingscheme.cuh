#ifndef WEIGHTINGSCHEME_CUH
#define WEIGHTINGSCHEME_CUH

#include <vector>

#include "io/MSFITSIO.cuh"
#include "uvtaper.cuh"

class WeightingScheme {
 public:
  virtual void apply(std::vector<MSDataset>& d) = 0;
  virtual void configure(void* params) = 0;

  WeightingScheme();
  WeightingScheme(int threads);
  WeightingScheme(int threads, UVTaper* uvtaper);
  WeightingScheme(int threads, UVTaper* uvtaper, bool modify_weights);

  bool getModifyWeights();
  void setModifyWeights(bool modify_weights);
  int getThreads();
  void setThreads(int threads);
  UVTaper* getUVTaper();
  void setUVTaper(UVTaper* uvtaper);
  void restoreWeights(std::vector<MSDataset>& d);

 protected:
  int threads;
  UVTaper* uvtaper = NULL;
  bool modify_weights;
};
#endif
