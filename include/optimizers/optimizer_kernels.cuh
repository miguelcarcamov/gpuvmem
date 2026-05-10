#ifndef OPTIMIZER_KERNELS_CUH
#define OPTIMIZER_KERNELS_CUH

#include <cuda_runtime.h>

__global__ void getGandDGG(float* gg, float* dgg, float* xi, float* g, long N);

__global__ void getGGandDGG(float* gg,
                            float* dgg,
                            float* xi,
                            float* g,
                            long N,
                            long M,
                            int image);

__global__ void searchDirection(float* g, float* xi, float* h, long N);

__global__ void searchDirection_LBFGS(float* xi, long N, long M, int image);

__global__ void normArray(float* result,
                          float* array,
                          int M,
                          int N,
                          int image);

__global__ void CGGradCondition(float* temp,
                                float* xi,
                                float* p,
                                float den,
                                int M,
                                int N,
                                int image);

__global__ void
updateQ(float* d_q, float alpha, float* d_y, int k, int M, int N, int image);

__global__ void getR(float* d_r,
                     float* d_q,
                     float scalar,
                     int M,
                     int N,
                     int image);

__global__ void calculateSandY(float* d_y,
                               float* d_s,
                               float* p,
                               float* xi,
                               float* p_old,
                               float* xi_old,
                               int iter,
                               int M,
                               int N,
                               int image);

/** Staged s,y (flat M*N*image + …) for curvature check before writing history slots. */
__global__ void calculateSandYScratch(float* scratch_y,
                                      float* scratch_s,
                                      float* p,
                                      float* xi,
                                      float* p_old,
                                      float* xi_old,
                                      int M,
                                      int N,
                                      int image);

__global__ void searchDirection(float* g,
                                float* xi,
                                float* h,
                                long N,
                                long M,
                                int image);

__global__ void newXi(float* g, float* xi, float* h, float gam, long N);

__global__ void newXi(float* g, float* xi, float* h, float gam, long N, long M, int image);

__global__ void
newXi(float* g, float* xi, float* h, float gam, long N, long M, int image);

__global__ void restartDPhi(float* dphi, float* dChi2, float* dH, long N);

// AddToDPhi moved to chi2/chi2_kernels.cuh (single-plane dgi version)

__global__ void getDot_LBFGS_ff(float* aux_vector,
                                float* vec_1,
                                float* vec_2,
                                int k,
                                int h,
                                int M,
                                int N,
                                int image);

#endif  // OPTIMIZER_KERNELS_CUH
