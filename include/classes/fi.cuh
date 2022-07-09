#ifndef FI_CUH
#define FI_CUH

#include <cuda_runtime.h>

#include "ckernel.cuh"

extern long M, N;
extern int image_count;
extern float* penalizators;
extern int nPenalizators;

class Fi {
 public:
  Fi::Fi() {
    this->name = "default";
    this->penalization_factor = 1.0f;
    this->Inu = NULL;
    this->iteration = 0;
  }

  virtual float calcFi(float* p) = 0;
  virtual void calcGi(float* p, float* xi) = 0;
  virtual float simulateModel(float* p){};
  virtual void restartDGi() = 0;
  virtual void addToDphi(float* device_dphi) = 0;
  virtual void setPrior(float prior){};
  virtual void setPrior(float* prior){};
  virtual float getEta(){};
  virtual void setEta(float eta){};
  virtual void setCKernel(CKernel* ckernel){};

  std::string getName() { return this->name; };

  std::string setName(std::string name) { this->name = name; };

  float get_fivalue() { return this->fi_value; };
  float getPenalizationFactor() { return this->penalization_factor; };
  void set_fivalue(float fi) { this->fi_value = fi; };
  void setPenalizationFactor(float p) { this->penalization_factor = p; };
  void setInu(cufftComplex* Inu) { this->Inu = Inu; }
  cufftComplex* getInu() { return this->Inu; }
  void setS(float* S) {
    cudaFree(device_S);
    this->device_S = S;
  };
  void setDS(float* DS) {
    cudaFree(device_DS);
    this->device_DS = DS;
  };
  void setIteration(int iteration) { this->iteration = iteration; };

  virtual float calculateSecondDerivate() = 0;
  virtual void configure(int penalizatorIndex, int imageIndex, int imageToAdd) {
    this->imageIndex = imageIndex;
    this->order = order;
    this->mod = mod;
    this->imageToAdd = imageToAdd;

    if (imageIndex > image_count - 1 || imageToAdd > image_count - 1) {
      printf("There is no image for the provided index %s\n", this->name);
      exit(-1);
    }

    if (penalizatorIndex != -1) {
      if (penalizatorIndex < 0) {
        printf("invalid index for penalizator (%s)\n", this->name);
        exit(-1);
      } else if (penalizatorIndex > (nPenalizators - 1)) {
        this->penalization_factor = 0.0f;
      } else {
        this->penalization_factor = penalizators[penalizatorIndex];
      }
    }

    checkCudaErrors(cudaMalloc((void**)&device_S, sizeof(float) * M * N));
    checkCudaErrors(cudaMemset(device_S, 0, sizeof(float) * M * N));

    checkCudaErrors(cudaMalloc((void**)&device_DS, sizeof(float) * M * N));
    checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
  };

 protected:
  float fi_value;
  float* device_S;
  float* device_DS;
  float penalization_factor;
  int imageIndex;
  int iteration;
  int mod;
  int order;
  std::string name;
  ;
  cufftComplex* Inu;
  int imageToAdd;
};

#endif
