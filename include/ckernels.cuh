#ifndef CKERNELS_CUH
#define CKERNELS_CUH
#include "framework.cuh"
#include "functions.cuh"

template <class T>
__host__ __device__ T EllipticalGaussian2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T angle);

template <class T>
__host__ __device__ T Gaussian2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T w, T alpha);

template <class T>
__host__ __device__ T Gaussian1D(T amp, T x, T x0, T sigma, T w, T alpha);

template <class T>
__host__ __device__ T Sinc1D(T amp, T x, T x0, T sigma, T w);

template <class T>
__host__ __device__ T GaussianSinc1D(T amp, T x, T x0, T sigma, T w1, T w2, T alpha);

template <class T>
__host__ __device__ T Sinc2D(T amp, T x, T x0, T y, T y0, T sigma_x, T sigma_y, T w);

template <class T>
__host__ __device__ T GaussianSinc2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T w1, T w2, T alpha);

template <class Tdx, class T>
class CKernel
{
public:
CKernel();
CKernel(Tdx dx, Tdx dy, int M, int N);
CKernel(Tdx dx, Tdx dy, T w1, T w2, T alpha, int M, int N);
T getdx();
T getdy();
int2 getMN();
T getW1();
T getW2();
T getAlpha();
T* getGPUKernel();
std::vector<T> getCPUKernel();
void setdxdy(Tdx dx, Tdx dy);
void setMN(int M, int N);
void setW1(T w1);
void setW2(T w2);
void setAlpha(T alpha);
void create2DKernelGPU(int id);
void create2DKernelCPU(int id);
private:
int M;
int N;
T w1;
T w2;
T alpha;
Tdx dx;
Tdx dy;
std::vector<T> kernel;
};

#endif
