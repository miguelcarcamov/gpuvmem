#ifndef FI_CUH
#define FI_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>

#include "ckernel.cuh"

class ObjectiveFunction;

class Fi {
 public:
  Fi::Fi() {
    this->name = "default";
    this->penalization_factor = 1.0f;
    this->Inu = NULL;
    this->iteration = 0;
    this->normalize = false;
  }

  virtual float calcFi(float* p) = 0;
  virtual void calcGi(float* p, float* xi) = 0;
  virtual void restartDGi() = 0;
  virtual void addToDphi(float* device_dphi) = 0;
  virtual void setPrior(float prior) {};
  virtual void setPrior(float* prior) {};
  virtual float getEta() {};
  virtual void setEta(float eta) {};
  virtual void setCKernel(CKernel* ckernel) {};
  virtual void setFgScale(float fg_scale) {};
  virtual float getFgScale() {};

  const std::string& getName() const { return this->name; }

  void setName(const std::string& name) { this->name = name; }

  float get_fivalue() { return this->fi_value; };
  bool getNormalize() { return this->normalize; };
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
  void setNormalize(bool normalize) { this->normalize = normalize; };

  /** Copy grid and -Z weight table pointer from the objective (call before configure). */
  void attachToObjectiveFunction(ObjectiveFunction* o);

  virtual float calculateSecondDerivate() = 0;
  virtual void configure(int penalizatorIndex,
                         int imageIndex,
                         int imageToAdd,
                         bool normalize) {
    this->imageIndex = imageIndex;
    this->order = imageIndex;
    this->mod = imageToAdd;
    this->imageToAdd = imageToAdd;
    this->normalize = normalize;

    if (grid_M_ <= 0 || grid_N_ <= 0 || grid_image_count_ <= 0) {
      std::cerr << "Fi::configure: image grid not set (attach objective to Image first). Term \""
                << this->name << "\"\n";
      exit(-1);
    }

    if (imageIndex > grid_image_count_ - 1 || imageToAdd > grid_image_count_ - 1) {
      std::cerr << "Fi::configure: image index out of range for term \"" << this->name << "\"\n";
      exit(-1);
    }

    if (penalizatorIndex != -1) {
      if (penalizatorIndex < 0) {
        std::cerr << "Fi::configure: invalid regularizer index for term \"" << this->name << "\"\n";
        exit(-1);
      } else if (n_z_weights_ <= 0 || penalizatorIndex > (n_z_weights_ - 1)) {
        this->penalization_factor = 0.0f;
      } else if (z_weights_ptr_ != nullptr) {
        this->penalization_factor = z_weights_ptr_[penalizatorIndex];
      } else {
        this->penalization_factor = 0.0f;
      }
    }

    const size_t plane = static_cast<size_t>(grid_M_) * static_cast<size_t>(grid_N_);
    checkCudaErrors(cudaMalloc((void**)&device_S, sizeof(float) * plane));
    checkCudaErrors(cudaMemset(device_S, 0, sizeof(float) * plane));

    checkCudaErrors(cudaMalloc((void**)&device_DS, sizeof(float) * plane));
    checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * plane));
  };

 protected:
  long gridM() const { return grid_M_; }
  long gridN() const { return grid_N_; }
  int gridImageCount() const { return grid_image_count_; }
  const float* zWeights() const { return z_weights_ptr_; }
  int zWeightCount() const { return n_z_weights_; }

 private:
  long grid_M_{0};
  long grid_N_{0};
  int grid_image_count_{0};
  const float* z_weights_ptr_{nullptr};
  int n_z_weights_{0};

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
  cufftComplex* Inu;
  int imageToAdd;
  bool normalize;
};

#endif
