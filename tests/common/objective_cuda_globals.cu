#include <cuda_runtime.h>

// Legacy globals used by regularizer / chi2 host code (normally in MFS).
long M = 8;
long N = 8;
int image_count = 1;
dim3 threadsPerBlockNN(16, 16, 1);
dim3 numBlocksNN(1, 1, 1);
int firstgpu = 0;
extern int flag_opt;
int iter = 1;
float noise_cut = 10.f;
float MINPIX = 0.f;
float eta = -1.f;
float* device_noise_image = nullptr;
int nPenalizators = 0;
float* penalizators = nullptr;
