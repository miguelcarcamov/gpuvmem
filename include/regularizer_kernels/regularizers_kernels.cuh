#ifndef REGULARIZERS_KERNELS_CUH
#define REGULARIZERS_KERNELS_CUH

#include <cuda_runtime.h>

__host__ __device__ float approxAbs(float val, float epsilon);

// L1 norm regularizers
__device__ float calculateL1norm(const float* __restrict__ I,
                                 float epsilon,
                                 float noise,
                                 float noise_cut,
                                 int index,
                                 int M,
                                 int N);

__global__ void L1Vector(float* __restrict__ L1,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float epsilon,
                         float noise_cut,
                         int index);

__device__ float calculateDNormL1(const float* __restrict__ I,
                                  float lambda,
                                  float noise,
                                  float epsilon,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N);

__global__ void DL1NormK(float* __restrict__ dL1,
                         const float* __restrict__ I,
                         const float* __restrict__ noise,
                         float epsilon,
                         float noise_cut,
                         float lambda,
                         long N,
                         long M,
                         int index);

// Generalized L1 norm regularizers
__device__ float calculateGL1norm(const float* __restrict__ I,
                                  float prior,
                                  float epsilon_a,
                                  float epsilon_b,
                                  float noise,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N);

__global__ void GL1Vector(float* __restrict__ L1,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          long N,
                          long M,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          int index);

__device__ float calculateDGNormL1(const float* __restrict__ I,
                                   float prior,
                                   float lambda,
                                   float noise,
                                   float epsilon_a,
                                   float epsilon_b,
                                   float noise_cut,
                                   int index,
                                   int M,
                                   int N);

__global__ void DGL1NormK(float* __restrict__ dL1,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          const float* __restrict__ noise,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          float lambda,
                          long N,
                          long M,
                          int index);

// Entropy regularizers
__device__ float calculateS(const float* __restrict__ I,
                            float G,
                            float eta,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N);

__device__ float calculateDS(const float* __restrict__ I,
                             float G,
                             float eta,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N);

__global__ void SVector(float* __restrict__ S,
                        const float* __restrict__ noise,
                        float* __restrict__ I,
                        long N,
                        long M,
                        float noise_cut,
                        float prior_value,
                        float eta,
                        int index);

__global__ void DS(float* __restrict__ dS,
                   float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   float prior_value,
                   float eta,
                   long N,
                   long M,
                   int index);

__global__ void SGVector(float* __restrict__ S,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         const float* __restrict__ prior,
                         float eta,
                         int index);

__global__ void DSG(float* __restrict__ dS,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float noise_cut,
                    float lambda,
                    const float* __restrict__ prior,
                    float eta,
                    long N,
                    long M,
                    int index);

// Quadratic prior regularizers
__device__ float calculateQP(const float* __restrict__ I,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N);

__global__ void QPVector(float* __restrict__ Q,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         int index);

__device__ float calculateDQ(const float* __restrict__ I,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N);

__global__ void DQ(float* __restrict__ dQ,
                   const float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index);

// L2 constant prior regularizers
__device__ float calculateL2ConstantPrior(const float* __restrict__ I,
                                          float prior,
                                          float noise,
                                          float noise_cut,
                                          int index,
                                          int M,
                                          int N);

__global__ void L2ConstantPriorVector(float* __restrict__ R,
                                      const float* __restrict__ noise,
                                      const float* __restrict__ I,
                                      float prior,
                                      long N,
                                      long M,
                                      float noise_cut,
                                      int index);

__device__ float calculateDL2ConstantPrior(const float* __restrict__ I,
                                           float prior,
                                           float lambda,
                                           float noise,
                                           float noise_cut,
                                           int index,
                                           int M,
                                           int N);

__global__ void DL2ConstantPriorKernel(float* __restrict__ dR,
                                       const float* __restrict__ I,
                                       const float* __restrict__ noise,
                                       float prior,
                                       float noise_cut,
                                       float lambda,
                                       long N,
                                       long M,
                                       int index);

// Laplacian regularizers
__device__ float calculateL(float* I,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N);

__global__ void LVector(float* L,
                        float* noise,
                        float* I,
                        long N,
                        long M,
                        float noise_cut,
                        int index);

__device__ float calculateDL(float* I,
                            float lambda,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N);

__global__ void DL(float* dL,
                   float* I,
                   float* noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index);

// Total Variation regularizers
__device__ float calculateTV(const float* __restrict__ I,
                             float epsilon,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N);

__global__ void TVVector(float* __restrict__ TV,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         float epsilon,
                         long N,
                         long M,
                         float noise_cut,
                         int index);

__device__ float calculateDTV(const float* __restrict__ I,
                              float epsilon,
                              float lambda,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N);

__global__ void DTV(float* __restrict__ dTV,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float epsilon,
                    float noise_cut,
                    float lambda,
                    long N,
                    long M,
                    int index);

// Anisotropic Total Variation regularizers
__device__ float calculateATV(const float* __restrict__ I,
                              float epsilon,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N);

__global__ void ATVVector(float* __restrict__ ATV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          float epsilon,
                          long N,
                          long M,
                          float noise_cut,
                          int index);

__device__ float calculateDATV(const float* __restrict__ I,
                               float epsilon,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N);

__global__ void DATV(float* __restrict__ dATV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float epsilon,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index);

// Total Squared Variation regularizers
__device__ float calculateTSV(const float* __restrict__ I,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N);

__global__ void TSVVector(float* __restrict__ STV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          long N,
                          long M,
                          float noise_cut,
                          int index);

__device__ float calculateDTSV(const float* __restrict__ I,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N);

__global__ void DTSV(float* __restrict__ dSTV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index);

#endif  // REGULARIZERS_KERNELS_CUH
