#ifndef ERROR_CUH
#define ERROR_CUH

#include "ms/ms_with_gpu.h"

#include <image.cuh>
#include <vector>

class Error {
 public:
  /** Compute error image from one or multiple datasets (group of MS + GPU). */
  virtual void calculateErrorImage(Image* I,
                                   std::vector<gpuvmem::ms::MSWithGPU>& datasets) = 0;
};

#endif  // ERROR_CUH
