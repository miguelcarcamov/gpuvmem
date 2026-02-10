#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <fcntl.h>
#include <float.h>
#include <getopt.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cufft.h>
#include <math_constants.h>
#include "device_launch_parameters.h"
#endif

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <ctgmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "flags.cuh"
#include "utils/constants.hh"  // For PI and PI_D constants

#ifdef __CUDACC__
#include "io/MSFITSIO.cuh"  // For CUDA-specific types and functions
#include <ckernel.cuh>
#include "utils/complexOps.cuh"
#include <error.cuh>
#include <factory.cuh>
#include <fi.cuh>
#include <filter.cuh>
#include <image.cuh>
#include <io.cuh>
#include <objectivefunction.cuh>
#include <optimizer.cuh>
#include <synthesizer.cuh>
#include <uvtaper.cuh>
#include <virtualimageprocessor.cuh>
#include <visibilities.cuh>
#include <weightingscheme.cuh>
#include "utils/copyrightwarranty.cuh"
#endif

// ============================================================================
// Constants and Enums
// ============================================================================

#define FLOAT_IMG -32
#define DOUBLE_IMG -64

#define TSTRING 16
#define TLONG 41
#define TINT 31
#define TFLOAT 42
#define TDOUBLE 82
#define TCOMPLEX 83
#define TDBLCOMPLEX 163

const float RPDEG = (PI / 180.0f);
const double RPDEG_D = (PI_D / 180.0);
const float RPARCSEC = (PI / (180.0f * 3600.0f));
const float RPARCSEC_D = (PI_D / (180.0 * 3600.0));
const float RPARCM = (PI / (180.0f * 60.0f));
const float RPARCM_D = (PI_D / (180.0 * 60.0));
const float RZ = 1.2196698912665045;

enum stokes {
  None,
  I_s,
  Q_s,
  U_s,
  V_s,
  RR,
  RL,
  LR,
  LL,
  XX,
  XY,
  YX,
  YY,
  RX,
  RY,
  LX,
  LY,
  XR,
  XL,
  YR,
  YL,
  PP,
  PQ,
  QP,
  QQ,
  RCircular,
  LCircular,
  Linear,
  Ptotal,
  Plinear,
  PFtotal,
  PFlinear,
  Pangle
};

extern long M, N;
extern int image_count;
extern float* penalizators;
extern int nPenalizators;

#ifdef __CUDACC__
typedef struct varsPerGPU {
  float* device_chi2;
  float* device_dchi2;
  cufftHandle plan;
  cufftComplex* device_I_nu;
  cufftComplex* device_V;
} varsPerGPU;
#endif

typedef struct variables {
  std::string input;
  std::string output;
  std::string inputdat;
  std::string modin;
  std::string ofile;
  std::string path;
  std::string output_image;
  std::string gpus;
  std::string initial_values;
  std::string penalization_factors;
  std::string user_mask;
  int blockSizeX;
  int blockSizeY;
  int blockSizeV;
  int it_max;
  int gridding;
  float noise;
  float noise_cut;
  float randoms;
  float eta;
  float nu_0;
  float robust_param;
  float threshold;
  float alpha_n_sigma;  // N sigma for alpha mask: alpha=0 where I_nu_0 <
                        // alpha_n_sigma*noise (e.g. 3 or 5)
  bool normalize;
  std::string optimization_mode;  // "joint" | "block" | "one" | "alpha_static"
  // joint = I_nu_0 + alpha together; block = alternate I_nu_0, alpha, ...;
  // one = single image / one run; alpha_static = two images, alpha fixed, I_nu_0 only
} Vars;

#ifdef __CUDACC__
class SynthesizerFactory {
 public:
  typedef Synthesizer* (*CreateSynthesizerCallback)();

 private:
  typedef std::map<int, CreateSynthesizerCallback> CallbackMap;

 public:
  // Returns true if registration was succesfull
  bool RegisterSynthesizer(int SynthesizerId,
                           CreateSynthesizerCallback CreateFn) {
    return callbacks_.insert(CallbackMap::value_type(SynthesizerId, CreateFn))
        .second;
  };

  bool UnregisterSynthesizer(int SynthesizerId) {
    return callbacks_.erase(SynthesizerId) == 1;
  };

  Synthesizer* CreateSynthesizer(int SynthesizerId) {
    CallbackMap::const_iterator i = callbacks_.find(SynthesizerId);
    if (i == callbacks_.end()) {
      // not found
      throw std::runtime_error("Unknown Synthesizer ID");
    }
    // Invoke the creation function
    return (i->second)();
  };

 private:
  CallbackMap callbacks_;
};

class WeightingSchemeFactory {
 public:
  typedef WeightingScheme* (*CreateWeightingSchemeCallback)();

 private:
  typedef std::map<int, CreateWeightingSchemeCallback> CallbackMap;

 public:
  // Returns true if registration was succesfull
  bool RegisterWeightingScheme(int WeightingSchemeId,
                               CreateWeightingSchemeCallback CreateFn) {
    return callbacks_
        .insert(CallbackMap::value_type(WeightingSchemeId, CreateFn))
        .second;
  };

  bool UnregisterWeightingScheme(int WeightingSchemeId) {
    return callbacks_.erase(WeightingSchemeId) == 1;
  };

  WeightingScheme* CreateWeightingScheme(int WeightingSchemeId) {
    CallbackMap::const_iterator i = callbacks_.find(WeightingSchemeId);
    if (i == callbacks_.end()) {
      // not found
      throw std::runtime_error("Unknown WeightingScheme ID");
    }
    // Invoke the creation function
    return (i->second)();
  };

 private:
  CallbackMap callbacks_;
};

class FiFactory {
 public:
  typedef Fi* (*CreateFiCallback)();

 private:
  typedef std::map<int, CreateFiCallback> CallbackMap;

 public:
  // Returns true if registration was succesfull
  bool RegisterFi(int FiId, CreateFiCallback CreateFn) {
    return callbacks_.insert(CallbackMap::value_type(FiId, CreateFn)).second;
  };

  bool UnregisterFi(int FiId) { return callbacks_.erase(FiId) == 1; };

  Fi* CreateFi(int FiId) {
    CallbackMap::const_iterator i = callbacks_.find(FiId);
    if (i == callbacks_.end()) {
      // not found
      throw std::runtime_error("Unknown Fi ID");
    }
    // Invoke the creation function
    return (i->second)();
  };

 private:
  CallbackMap callbacks_;
};

class OptimizatorFactory {
 public:
  typedef Optimizer* (*CreateOptimizatorCallback)();

 private:
  typedef std::map<int, CreateOptimizatorCallback> CallbackMap;

 public:
  // Returns true if registration was succesfull
  bool RegisterOptimizator(int OptimizatorId,
                           CreateOptimizatorCallback CreateFn) {
    return callbacks_.insert(CallbackMap::value_type(OptimizatorId, CreateFn))
        .second;
  };

  bool UnregisterOptimizator(int OptimizatorId) {
    return callbacks_.erase(OptimizatorId) == 1;
  };

  Optimizer* CreateOptimizator(int OptimizatorId) {
    CallbackMap::const_iterator i = callbacks_.find(OptimizatorId);
    if (i == callbacks_.end()) {
      // not found
      throw std::runtime_error("Unknown optimizer ID");
    }
    // Invoke the creation function
    return (i->second)();
  };

 private:
  CallbackMap callbacks_;
};

class CKernelFactory {
 public:
  typedef CKernel* (*CreateCKernelCallback)();

 private:
  typedef std::map<int, CreateCKernelCallback> CallbackMap;

 public:
  // Returns true if registration was succesfull
  bool RegisterCKernel(int CKernelId, CreateCKernelCallback CreateFn) {
    return callbacks_.insert(CallbackMap::value_type(CKernelId, CreateFn))
        .second;
  };

  bool UnregisterCKernel(int CKernelId) {
    return callbacks_.erase(CKernelId) == 1;
  };

  CKernel* CreateCKernel(int CKernelId) {
    CallbackMap::const_iterator i = callbacks_.find(CKernelId);
    if (i == callbacks_.end()) {
      // not found
      throw std::runtime_error("Unknown CKernel ID");
    }
    // Invoke the creation function
    return (i->second)();
  };

 private:
  CallbackMap callbacks_;
};

#endif  // __CUDACC__

#endif  // FRAMEWORK_CUH
