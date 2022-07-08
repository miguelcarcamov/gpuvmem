#ifndef PSWF_12D_CUH
#define PSWF_12D_CUH

#include "framework.cuh"
#include "functions.cuh"

__host__ float pswf_11D_func(float nu);
__host__ float pswf_11D(float amp, float x, float x0, float sigma, float w);
__host__ float pswf_12D(float amp, float x, float y, float x0, float y0,
                        float sigma_x, float sigma_y, float w);
class PSWF_12D : public CKernel {
 public:
  __host__ PSWF_12D();
  __host__ PSWF_12D(int m, int n);
  __host__ PSWF_12D(int m, int n, float w);
  __host__ PSWF_12D(int m, int n, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, Io *imageHandler);
  __host__ PSWF_12D(int m, int n, Io *imageHandler, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float dx, float dy);
  __host__ PSWF_12D(int m, int n, float dx, float dy, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float dx, float dy, Io *imageHandler);
  __host__ PSWF_12D(int m, int n, float dx, float dy, Io *imageHandler,
                    CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float w, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float w, Io *imageHandler, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float dx, float dy, float w);
  __host__ PSWF_12D(int m, int n, float dx, float dy, float w, CKernel *gcf);
  __host__ PSWF_12D(int m, int n, float dx, float dy, float w,
                    Io *imageHandler);
  __host__ PSWF_12D(int m, int n, float dx, float dy, float w, Io *imageHandler,
                    CKernel *gcf);
  __host__ void buildKernel(float amp, float x0, float y0, float sigma_x,
                            float sigma_y);
  __host__ void buildKernel();
  __host__ float GCF(float amp, float x, float y, float x0, float y0,
                     float sigma_x, float sigma_y, float w, float alpha);
  __host__ void buildGCF(float amp, float x0, float y0, float sigma_x,
                         float sigma_y) override;
  __host__ void buildGCF() override;
  __host__ CKernel *clone() const;

 private:
  __host__ void nameSelf();
};

#endif
