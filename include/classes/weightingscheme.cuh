#ifndef WEIGHTINGSCHEME_CUH
#define WEIGHTINGSCHEME_CUH

#include "ms/ms_with_gpu.h"
#include "uvtaper.cuh"

#include <vector>

class WeightingScheme {
 public:
  virtual void apply(std::vector<gpuvmem::ms::MSWithGPU>& d) = 0;
  virtual void configure(void* params) = 0;

  WeightingScheme() {
    this->threads = omp_get_num_procs() - 2;
    this->uvtaper = NULL;
    this->modify_weights = false;
  };

  WeightingScheme(int threads) {
    this->threads = threads;
    this->uvtaper = NULL;
    this->modify_weights = false;
  };

  WeightingScheme(int threads, UVTaper* uvtaper) {
    this->threads = threads;
    this->uvtaper = uvtaper;
    this->modify_weights = false;
  };

  WeightingScheme(int threads, UVTaper* uvtaper, bool modify_weights) {
    this->threads = threads;
    this->uvtaper = uvtaper;
    this->modify_weights = modify_weights;
  };

  bool getModifyWeights() { return this->modify_weights; };

  void setModifyWeights(bool modify_weights) {
    this->modify_weights = modify_weights;
  };

  int getThreads() { return this->threads; };

  void setThreads(int threads) {
    this->threads = threads;
    std::cout << "The running weighting scheme threads have been set to "
              << this->threads << std::endl;
  };

  UVTaper* getUVTaper() { return this->uvtaper; };

  void setUVTaper(UVTaper* uvtaper) {
    this->uvtaper = uvtaper;
    std::cout << "UVTaper has been set" << std::endl;
    std::cout << "UVTaper Features - bmaj=" << this->uvtaper->getSigma_maj()
              << ", bmin=" << this->uvtaper->getSigma_min()
              << ", bpa=" << this->uvtaper->getBPA() << std::endl;
  };

  /** Reset imaging_weight = weight for all visibilities (restore from MS values). */
  void restoreWeights(std::vector<gpuvmem::ms::MSWithGPU>& d) {
    for (auto& dw : d) {
      for (size_t f = 0; f < dw.ms.num_fields(); f++) {
        gpuvmem::ms::Field& field = dw.ms.field(f);
        for (auto& bl : field.baselines()) {
          for (auto& ts : bl.time_samples()) {
            ts.restore_imaging_weights();
          }
        }
      }
    }
  }

 protected:
  int threads;
  UVTaper* uvtaper = NULL;
  bool modify_weights;
};
#endif
