#ifndef ALPHAMFS_CUH
#define ALPHAMFS_CUH

#include <time.h>

#include <measures/Measures.h>
#include <measures/Measures/MDirection.h>
#include "utils/direction_cosines.cuh"
#include "framework.cuh"
#include "classes/imaging_header.hh"
#include "optimizers/conjugategradient.cuh"
#include "framework.cuh"

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
  __host__ void configure(const GpuvmemCliConfig& config) override;
  void applyFilter(Filter* filter) {
    if (this->getDatasets())
      filter->applyCriteria(*this->getDatasets());
  }

  const GpuvmemCliConfig& cliConfig() const { return cli_config_; }

 protected:
  std::vector<float> minimal_pixel_values;  // Store minimal pixel values for Image object

 private:
  GpuvmemCliConfig cli_config_;
  /** Model image astrometry (in-memory, 0-based reference pixel) copied onto Image in setDevice(). */
  gpuvmem::ImagingHeader resolved_model_header_{};
  void syncLegacyGlobalsFromCli_(const GpuvmemCliConfig& cfg);
};

#endif
