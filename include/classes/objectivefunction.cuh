#ifndef OBJECTIVEFUNCTION_CUH
#define OBJECTIVEFUNCTION_CUH

#include "fi.cuh"
#include "io.cuh"
#include "error.cuh"
#include "factory.cuh"
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

class ObjectiveFunction {
 public:
  ObjectiveFunction(){};
  /** Skip terms with λ==0 so their calcGi/addToDphi are never called (saves work). */
  void addFi(Fi* fi) {
    if (fi == nullptr) return;
    fi->attachToObjectiveFunction(this);
    if (fi->getPenalizationFactor()) {
      fis.push_back(fi);
      fi_values.push_back(0.0f);
    }
  };
  float calcFunction(float* p) {
    float value = 0.0;
    int fi_value_count = 0;
    for (std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++) {
      float iterationValue = (*it)->calcFi(p);
      fi_values[fi_value_count] = (*it)->get_fivalue();
      value += iterationValue;
      fi_value_count++;
    }

    return value;
  };

  void calcGradient(float* p, float* xi, int iter) {
    // Ensure dphi is allocated (configure must be called first)
    if (dphi == nullptr) {
      std::cerr << "ERROR: calcGradient called but dphi is not allocated. configure() must be called first!" << std::endl;
      return;
    }
    if (xi == nullptr) {
      std::cerr << "ERROR: calcGradient called with null xi pointer!" << std::endl;
      return;
    }
    if (this->M <= 0 || this->N <= 0 || this->image_count <= 0) {
      std::cerr << "ERROR: calcGradient called with invalid dimensions: M=" << this->M 
                << " N=" << this->N << " image_count=" << this->image_count << std::endl;
      return;
    }
    if (io != nullptr && io->getPrintImages()) {
      if (IoOrderIterations == NULL) {
        io->printImageIteration(p, "I_nu_0", "JY/PIXEL", iter, 0, true);
        io->printImageIteration(p, "alpha", "JY/PIXEL", iter, 1, true);
      } else {
        (IoOrderIterations)(p, io);
      }
    }
    restartDPhi();
    
    for (std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++) {
      (*it)->setIteration(iter);
      (*it)->calcGi(p, xi);
      (*it)->addToDphi(dphi);
    }
    // Note: linkAddToDPhi sets device to firstgpu, and we stay on that device
    // for copyDphiToXi (original behavior - dphi was modified on firstgpu)
    phiStatus = 1;
    copyDphiToXi(xi);
  };

  void restartDPhi() {
    if (dphi == nullptr) {
      // dphi not initialized - configure() must be called first
      return;
    }
    if (this->M <= 0 || this->N <= 0 || this->image_count <= 0) {
      // Invalid dimensions
      return;
    }
    // Zero each term's gradient buffer (e.g. Chi2 result_dchi2 for all images)
    for (std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++) {
      (*it)->restartDGi();
    }
    // Zero full dphi (I_nu_0 and alpha slices) so each term adds into a clean buffer
    checkCudaErrors(cudaMemset(dphi, 0, sizeof(float) * this->M * this->N * this->image_count));
  }

  void copyDphiToXi(float* xi) {
    // Original simple implementation - just copy device-to-device
    // The device context should already be set correctly by the caller
    checkCudaErrors(cudaMemcpy(xi, dphi, sizeof(float) * this->M * this->N * this->image_count,
                               cudaMemcpyDeviceToDevice));
  }
  
  // Get current gradient (dphi) for line searchers to access
  // dphi contains the gradient computed by calcGradient
  float* getCurrentGradient() { return dphi; }

  std::vector<Fi*> getFi() { return this->fis; };

  Fi* getFiByName(std::string fi_name) {
    Fi* found_fi = NULL;
    for (std::vector<Fi*>::iterator it = this->fis.begin();
         it != this->fis.end(); it++) {
      if ((*it)->getName() == fi_name) {
        found_fi = (*it);
        break;
      }
    }
    return found_fi;
  };

  void setN(long N) { this->N = N; }
  void setM(long M) { this->M = M; }
  void setImageCount(int I) { this->image_count = I; }

  /** Set grid dimensions only (no dphi allocation). Used before Fi::configure in the driver. */
  void setGridDimensions(long N, long M, int image_count_in) {
    this->N = N;
    this->M = M;
    this->image_count = image_count_in;
  }

  /** Non-owning pointer to the -Z weight list (same lifetime as MFS penalty storage). */
  void setRegularizationWeights(const float* weights, int n) {
    reg_weights_ptr_ = weights;
    n_reg_weights_ = n;
  }
  const float* getRegularizationWeights() const { return reg_weights_ptr_; }
  int getRegularizationWeightCount() const { return n_reg_weights_; }

  // Getters for dimensions (used by seeders and line searchers)
  long getM() const { return this->M; }
  long getN() const { return this->N; }
  int getImageCount() const { return this->image_count; }
  
  // CUDA launch configuration (computed from M, N)
  void setThreadsPerBlockNN(dim3 threads) { this->threadsPerBlockNN = threads; }
  void setNumBlocksNN(dim3 blocks) { this->numBlocksNN = blocks; }
  dim3 getThreadsPerBlockNN() const { return this->threadsPerBlockNN; }
  dim3 getNumBlocksNN() const { return this->numBlocksNN; }
  void setIo(Io* i) { this->io = i; };

  /** Primary CUDA device for line-search / gradient kernels (replaces `extern firstgpu` there). */
  void setPrimaryCudaDevice(int device) { primary_cuda_device_ = device; }
  int getPrimaryCudaDevice() const { return primary_cuda_device_; }

  void setIoOrderIterations(void (*func)(float* I, Io* io)) {
    this->IoOrderIterations = func;
  };
  void configure(long N, long M, int I) {
    const bool need_alloc =
        (dphi == nullptr || this->N != N || this->M != M || this->image_count != I);
    setGridDimensions(N, M, I);
    if (!need_alloc) {
      return;
    }
    if (dphi != nullptr) {
      cudaFree(dphi);
      dphi = nullptr;
    }
    if (this->M <= 0 || this->N <= 0 || this->image_count <= 0) {
      std::cerr << "ERROR: configure() called with invalid dimensions: M=" << this->M
                << " N=" << this->N << " image_count=" << this->image_count << std::endl;
      return;
    }
    size_t alloc_size = sizeof(float) * this->M * this->N * this->image_count;
    checkCudaErrors(cudaMalloc((void**)&dphi, alloc_size));
    checkCudaErrors(cudaMemset(dphi, 0, alloc_size));
  }
  std::vector<float> get_fi_values() { return this->fi_values; }

 private:
  std::vector<Fi*> fis;
  std::vector<float> fi_values;
  Io* io = NULL;
  float* dphi = nullptr;  // Current gradient (computed by calcGradient, accessible via getCurrentGradient)
  int phiStatus = 1;
  int flag = 0;
  long N = 0;
  long M = 0;
  void (*IoOrderIterations)(float* I, Io* io) = NULL;
  int image_count = 1;
  dim3 threadsPerBlockNN = dim3(0, 0, 0);  // CUDA launch configuration
  dim3 numBlocksNN = dim3(0, 0, 0);        // CUDA launch configuration
  int primary_cuda_device_{0};
  const float* reg_weights_ptr_{nullptr};
  int n_reg_weights_{0};
};

namespace {
ObjectiveFunction* CreateObjectiveFunction() {
  return new ObjectiveFunction;
}
const std::string ObjectiveFunctionId = "ObjectiveFunction";
const bool RegisteredObjectiveFunction =
    registerCreationFunction<ObjectiveFunction, std::string>(
        ObjectiveFunctionId,
        CreateObjectiveFunction);
};  // namespace

#endif  // OBJECTIVEFUNCTION_CUH
