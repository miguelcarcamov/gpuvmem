#ifndef FILTER_CUH
#define FILTER_CUH

#include "ms/ms_with_gpu.h"

#include <vector>

class Filter {
 public:
  /** Apply filter to one or multiple datasets (group of MS + GPU). */
  virtual void applyCriteria(std::vector<gpuvmem::ms::MSWithGPU>& datasets) = 0;
  virtual void configure(void* params) = 0;
};

#endif  // FILTER_CUH
