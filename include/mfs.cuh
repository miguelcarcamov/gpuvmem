#ifndef ALPHAMFS_CUH
#define ALPHAMFS_CUH

#include <time.h>

#include <measures/Measures.h>
#include <measures/Measures/MDirection.h>
#include "directioncosines.cuh"
#include "framework.cuh"
#include "optimizers/conjugategradient.cuh"
#include "functions.cuh"

class MFS : public Synthesizer {
 public:
  void writeImages();
  void clearRun();
  void writeResiduals();
  void run();
  void setOutPut(char* FileName){};
  void setDevice();
  void unSetDevice();
  std::vector<std::string> countAndSeparateStrings(std::string long_str,
                                                   std::string sep);
  void configure(int argc, char** argv);
  void applyFilter(Filter* filter) {
    if (this->getDatasets())
      filter->applyCriteria(*this->getDatasets());
  }
  
 protected:
  std::vector<float> minimal_pixel_values;  // Store minimal pixel values for Image object
};

#endif
