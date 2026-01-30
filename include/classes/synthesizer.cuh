#ifndef SYNTHESIZER_CUH
#define SYNTHESIZER_CUH

#include "ms/ms_with_gpu.h"
#include "weightingscheme.cuh"

#include <vector>

class Synthesizer {
 public:
  __host__ virtual void run() = 0;
  __host__ virtual void setOutPut(char* FileName) = 0;
  __host__ virtual void setDevice() = 0;
  __host__ virtual void unSetDevice() = 0;
  __host__ virtual std::vector<std::string> countAndSeparateStrings(
      std::string long_str,
      std::string sep) = 0;
  __host__ virtual void configure(int argc, char** argv) = 0;
  __host__ virtual void applyFilter(Filter* filter) = 0;
  __host__ virtual void writeImages() = 0;
  __host__ virtual void clearRun() = 0;
  __host__ virtual void writeResiduals() = 0;
  __host__ void setOptimizator(Optimizer* min) { this->optimizer = min; };
  /** Set the group of datasets (one or multiple MS + GPU). Pipeline owns the vector. */
  __host__ void setDatasets(std::vector<gpuvmem::ms::MSWithGPU>* d) {
    datasets_ = d;
  }
  __host__ std::vector<gpuvmem::ms::MSWithGPU>* getDatasets() {
    return datasets_;
  }
  __host__ void setTotalVisibilities(int t) { total_visibilities_ = t; }
  __host__ void setNDatasets(int n) { ndatasets_ = n; }
  __host__ void setMaxNumberVis(int m) { max_number_vis_ = m; }
  __host__ int getTotalVisibilities() const { return total_visibilities_; }
  __host__ int getNDatasets() const { return ndatasets_; }
  __host__ int getMaxNumberVis() const { return max_number_vis_; }

  __host__ void setIoImageHandler(Io* imageHandler) {
    this->ioImageHandler = imageHandler;
  };

  __host__ void setIoVisibilitiesHandler(Io* visHandler) {
    this->ioVisibilitiesHandler = visHandler;
  };

  __host__ void setError(Error* e) { this->error = e; };

  __host__ void setWeightingScheme(WeightingScheme* scheme) {
    this->scheme = scheme;
  };
  __host__ void setOrder(void (*func)(Optimizer* o, Image* I)) {
    this->Order = func;
  };
  Image* getImage() { return image; };
  void setImage(Image* i) { this->image = i; };
  void setIoOrderEnd(void (*func)(float* I, Io* io)) {
    this->IoOrderEnd = func;
  };
  void setIoOrderError(void (*func)(float* I, Io* io)) {
    this->IoOrderError = func;
  };
  void setIoOrderIterations(void (*func)(float* I, Io* io)) {
    this->IoOrderIterations = func;
  };

  Optimizer* getOptimizator() { return this->optimizer; };

  __host__ void setGriddingKernel(CKernel* ckernel) {
    this->ckernel = ckernel;
  };

  __host__ bool getGridding() { return this->gridding; };

  __host__ void setGridding(bool gridding) { this->gridding = gridding; };

  __host__ int getGriddingThreads() { return this->griddingThreads; };

  __host__ void setGriddingThreads(int griddingThreads) {
    if (griddingThreads > 0) {
      this->griddingThreads = griddingThreads;
      this->gridding = true;
    } else {
      std::cout << "Gridding threads cannot be less than 0" << std::endl;
    }
  };

  __host__ float getVisNoise() { return this->vis_noise; };

  __host__ void setVisNoise(float noise) { this->vis_noise = noise; };

  __host__ float getFgScale() { return this->fg_scale; };

  __host__ void setFgScale(float fg_scale) { this->fg_scale = fg_scale; };

 protected:
  cufftComplex* device_I;
  Image* image;
  Optimizer* optimizer;
  CKernel* ckernel;
  Io* ioImageHandler = NULL;
  Io* ioVisibilitiesHandler = NULL;
  std::vector<gpuvmem::ms::MSWithGPU>* datasets_ = nullptr;
  int total_visibilities_ = 0;
  int ndatasets_ = 0;
  int max_number_vis_ = 0;
  Error* error = NULL;
  int griddingThreads = 0;
  bool gridding = false;
  float fg_scale = 1.0;
  float vis_noise = 0.0;
  void (*Order)(Optimizer* o, Image* I) = NULL;
  int imagesChanged = 0;
  void (*IoOrderIterations)(float* I, Io* io) = NULL;
  void (*IoOrderEnd)(float* I, Io* io) = NULL;
  void (*IoOrderError)(float* I, Io* io) = NULL;
  WeightingScheme* scheme = NULL;
};
#endif  // GPUVMEM_SYNTHESIZER_CUH
