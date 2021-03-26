#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <ctgmath>
#include <string>
#include <functional>
#include <numeric>
#include <cstdint>
#include "device_launch_parameters.h"
#include <cufft.h>
#include <math_constants.h>
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include "copyrightwarranty.cuh"
#include <cooperative_groups.h>
#include <weightingscheme.cuh>
#include <image.cuh>
#include <visibilities.cuh>
#include <error.cuh>
#include <fi.cuh>
#include <virtualimageprocessor.cuh>
#include <io.cuh>
#include <objectivefunction.cuh>
#include <optimizer.cuh>
#include <filter.cuh>
#include <ckernel.cuh>
#include <uvtaper.cuh>
#include <synthesizer.cuh>
#include <factory.cuh>
#include <flags.cuh>
#include <complexOps.cuh>
#include <boost/algorithm/string.hpp>

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

typedef struct varsPerGPU {
        float *device_chi2;
        float *device_dchi2;
        cufftHandle plan;
        cufftComplex *device_I_nu;
        cufftComplex *device_V;
}varsPerGPU;

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
} Vars;

class SynthesizerFactory
{
public:
typedef Synthesizer* (*CreateSynthesizerCallback)();
private:
typedef std::map<int, CreateSynthesizerCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterSynthesizer(int SynthesizerId, CreateSynthesizerCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(SynthesizerId, CreateFn)).second;
};

bool UnregisterSynthesizer(int SynthesizerId)
{
        return callbacks_.erase(SynthesizerId) == 1;
};

Synthesizer* CreateSynthesizer(int SynthesizerId)
{
        CallbackMap::const_iterator i = callbacks_.find(SynthesizerId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Synthesizer ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class WeightingSchemeFactory
{
public:
typedef WeightingScheme* (*CreateWeightingSchemeCallback)();
private:
typedef std::map<int, CreateWeightingSchemeCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterWeightingScheme(int WeightingSchemeId, CreateWeightingSchemeCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(WeightingSchemeId, CreateFn)).second;
};

bool UnregisterWeightingScheme(int WeightingSchemeId)
{
        return callbacks_.erase(WeightingSchemeId) == 1;
};

WeightingScheme* CreateWeightingScheme(int WeightingSchemeId)
{
        CallbackMap::const_iterator i = callbacks_.find(WeightingSchemeId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown WeightingScheme ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class FiFactory
{
public:
typedef Fi* (*CreateFiCallback)();
private:
typedef std::map<int, CreateFiCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterFi(int FiId, CreateFiCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(FiId, CreateFn)).second;
};

bool UnregisterFi(int FiId)
{
        return callbacks_.erase(FiId) == 1;
};

Fi* CreateFi(int FiId)
{
        CallbackMap::const_iterator i = callbacks_.find(FiId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Fi ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class OptimizatorFactory
{
public:
typedef Optimizer* (*CreateOptimizatorCallback)();
private:
typedef std::map<int, CreateOptimizatorCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterOptimizator(int OptimizatorId, CreateOptimizatorCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(OptimizatorId, CreateFn)).second;
};

bool UnregisterOptimizator(int OptimizatorId)
{
        return callbacks_.erase(OptimizatorId) == 1;
};

Optimizer* CreateOptimizator(int OptimizatorId)
{
        CallbackMap::const_iterator i = callbacks_.find(OptimizatorId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown optimizer ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class CKernelFactory
{
public:
typedef CKernel* (*CreateCKernelCallback)();
private:
typedef std::map<int, CreateCKernelCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterCKernel(int CKernelId, CreateCKernelCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(CKernelId, CreateFn)).second;
};

bool UnregisterCKernel(int CKernelId)
{
        return callbacks_.erase(CKernelId) == 1;
};

CKernel* CreateCKernel(int CKernelId)
{
        CallbackMap::const_iterator i = callbacks_.find(CKernelId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown CKernel ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

#endif
