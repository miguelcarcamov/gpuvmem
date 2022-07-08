#ifndef SINC2D_CUH
#define SINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

__host__ float sincf(float x);
__host__ float sinc1D(float amp, float x, float x0, float sigma, float w);
__host__ float sinc2D(float amp, float x, float x0, float y, float y0,
                      float sigma_x, float sigma_y, float w);
class Sinc2D : public CKernel {
 public:
  __host__ Sinc2D();
  __host__ Sinc2D(int m, int n);
  __host__ Sinc2D(int m, int n, float w);
  __host__ Sinc2D(int m, int n, CKernel *gcf);
  __host__ Sinc2D(int m, int n, Io *imageHandler);
  __host__ Sinc2D(int m, int n, Io *imageHandler, CKernel *gcf);
  __host__ Sinc2D(int m, int n, float dx, float dy);
  __host__ Sinc2D(int m, int n, float dx, float dy, CKernel *gcf);
  __host__ Sinc2D(int m, int n, float dx, float dy, Io *imageHandler);
  __host__ Sinc2D(int m, int n, float dx, float dy, Io *imageHandler,
                  CKernel *gcf);
  __host__ Sinc2D(int m, int n, float w, CKernel *gcf);
  __host__ Sinc2D(int m, int n, float w, Io *imageHandler, CKernel *gcf);
  __host__ Sinc2D(int m, int n, float dx, float dy, float w);
  __host__ Sinc2D(int m, int n, float dx, float dy, float w, CKernel *gcf);
  __host__ Sinc2D(int m, int n, float dx, float dy, float w, Io *imageHandler);
  __host__ Sinc2D(int m, int n, float dx, float dy, float w, Io *imageHandler,
                  CKernel *gcf);
  __host__ void buildKernel();
  __host__ void buildKernel(float amp, float x0, float y0, float sigma_x,
                            float sigma_y);
  __host__ float GCF_fn(float amp, float nu, float w);
  __host__ float GCF(float amp, float x, float y, float x0, float y0,
                     float sigma_x, float sigma_y, float w, float alpha);
  __host__ CKernel *clone() const;

 private:
  __host__ void nameSelf();
};

#endif
