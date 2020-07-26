#ifndef CKERNELS_CUH
#define CKERNELS_CUH
#include "framework.cuh"
#include "functions.cuh"

class CKernel
{
public:
CKernel();
CKernel(float dx, float dy, int M, int N);
CKernel(float dx, float dy, float w1, float w2, float alpha, int M, int N);
float getdx();
float getdy();
int2 getMN();
float getW1();
float getW2();
float getAlpha();
float* getGPUKernel();
std::vector<float> getCPUKernel();
void setdxdy(float dx, float dy);
void setMN(int M, int N);
void setW1(float w1);
void setW2(float w2);
void setAlpha(float alpha);
void create2DKernelGPU(int id);
void create2DKernelCPU(int id);
float run(float deltau, float deltav){return 1.0f;};
void setExecutionMode(int mode){this->mode = mode;};
private:
float EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
float Gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha);
float Gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha);
float Sinc1D(float amp, float x, float x0, float sigma, float w);
float GaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha);
float Sinc2D(float amp, float x, float x0, float y, float y0, float sigma_x, float sigma_y, float w);
float GaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha);
int M;
int N;
float w1;
float w2;
float alpha;
float dx;
float dy;
int mode;
std::vector<float> kernel;
};

#endif
