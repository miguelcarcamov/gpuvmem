#ifndef LBFGS_CUH
#define LBFGS_CUH
#include "linmin.cuh"

class LBFGS : public Optimizer {
 public:
  __host__ int getK() override;
  __host__ void setK(int K) override;
  __host__ void allocateMemoryGpu();
  __host__ void deallocateMemoryGpu();
  __host__ void optimize();
  __host__ void LBFGS_recursion(float *d_y, float *d_s, float *xi, int par_M,
                                int lbfgs_it, int M, int N);

 private:
  float fret = 0;
  float gg, dgg, gam, fp;
  float *device_g, *device_h;
  float *device_gg_vector, *device_dgg_vector;
  int configured = 1;
  float *d_s;
  float *d_y, *xi, *xi_old, *norm_vector, *d_r;
  float max_per_it;
  float *p_old;
  int K = 100;
};

#endif
