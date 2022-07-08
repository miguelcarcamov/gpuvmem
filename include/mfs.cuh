#ifndef ALPHAMFS_CUH
#define ALPHAMFS_CUH

#include <time.h>

#include "directioncosines.cuh"
#include "framework.cuh"
#include "frprmn.cuh"
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
    filter->applyCriteria(this->visibilities);
  };
};

#endif
