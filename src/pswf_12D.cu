#include "pswf_12D.cuh"

__host__ float pswf_11D_func(float nu) {
  float nu_end;
  float dnusq, top, bottom;
  int idx;

  const float mat_p[2][5] = {
      {8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
      {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};

  const float mat_q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                             {1.0000000e0, 9.599102e-1, 2.918724e-1}};

  float n_nu = fabsf(nu);
  float res = 0.0f;
  if (n_nu > 1.0f)
    res = 0.0f;
  else {
    nu_end = 0.0f;
    idx = 0;
    if (n_nu >= 0.0f && n_nu < 0.75) {
      idx = 0;
      nu_end = 0.75f;
    } else {
      idx = 1;
      nu_end = 1.0f;
    }

    dnusq = n_nu * n_nu - nu_end * nu_end;
    top = mat_p[idx][0];
    bottom = mat_q[idx][0];

    for (int i = 1; i < 5; i++) {
      top += mat_p[idx][i] * powf(dnusq, i);
    }

    for (int i = 1; i < 3; i++) {
      bottom += mat_q[idx][i] * powf(dnusq, i);
    }

    if (bottom > 0.0f) {
      res = top / bottom;
    }
  }
  return res;
};

__host__ float pswf_11D(float amp, float x, float x0, float sigma, float w) {
  float nu, pswf, nu_sq, val;
  float radius = distance(x, 0.0f, x0, 0.0f);

  nu = radius / (w * sigma);
  if (nu == 0.0f) {
    val = 1.0f;
  } else {
    pswf = pswf_11D_func(nu);
    nu_sq = nu * nu;
    val = amp * (1.0f - nu_sq) * pswf;
  }
  return val;
};

__host__ float pswf_12D(float amp,
                        float x,
                        float y,
                        float x0,
                        float y0,
                        float sigma_x,
                        float sigma_y,
                        float w) {
  float xval = pswf_11D(1.0f, x, x0, sigma_x, w);
  float yval = pswf_11D(1.0f, y, y0, sigma_y, w);
  float val = amp * xval * yval;
  return val;
};

__host__ PSWF_12D::PSWF_12D() : CKernel() {
  this->w = 6.0f;
};

__host__ PSWF_12D::PSWF_12D(int m, int n) : CKernel(m, n) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float w) : CKernel(m, n, w) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, CKernel* gcf) : CKernel(m, n, gcf) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, Io* imageHandler)
    : CKernel(m, n, imageHandler) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, Io* imageHandler, CKernel* gcf)
    : CKernel(m, n, imageHandler, gcf) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float dx, float dy)
    : CKernel(m, n, dx, dy) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float dx, float dy, CKernel* gcf)
    : CKernel(m, n, dx, dy, gcf) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float dx, float dy, Io* imageHandler)
    : CKernel(m, n, dx, dy, imageHandler) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m,
                            int n,
                            float dx,
                            float dy,
                            Io* imageHandler,
                            CKernel* gcf)
    : CKernel(m, n, dx, dy, imageHandler, gcf) {
  this->w = 6.0f;
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float w, CKernel* gcf)
    : CKernel(m, n, w, gcf) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m,
                            int n,
                            float w,
                            Io* imageHandler,
                            CKernel* gcf)
    : CKernel(m, n, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m, int n, float dx, float dy, float w)
    : CKernel(m, n, dx, dy, w) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m,
                            int n,
                            float dx,
                            float dy,
                            float w,
                            CKernel* gcf)
    : CKernel(m, n, dx, dy, w, gcf) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m,
                            int n,
                            float dx,
                            float dy,
                            float w,
                            Io* imageHandler)
    : CKernel(m, n, dx, dy, w, imageHandler) {
  this->nameSelf();
};

__host__ PSWF_12D::PSWF_12D(int m,
                            int n,
                            float dx,
                            float dy,
                            float w,
                            Io* imageHandler,
                            CKernel* gcf)
    : CKernel(m, n, dx, dy, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ void PSWF_12D::buildKernel(float amp,
                                    float x0,
                                    float y0,
                                    float sigma_x,
                                    float sigma_y) {
  this->setKernelMemory();
  float x, y;
  float val;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w);
      this->kernel[this->n * i + j] = val;
    }
  }
  this->copyKerneltoGPU();
  if (NULL != this->gcf) {
  }
};

__host__ void PSWF_12D::buildKernel() {
  this->setKernelMemory();
  float x, y;
  float val;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      val = pswf_12D(this->amp, x, y, this->x0, this->y0, this->sigma_x,
                     this->sigma_y, this->w);
      this->kernel[this->n * i + j] = val;
    }
  }
  this->copyKerneltoGPU();
  if (NULL != this->gcf) {
  }
};

__host__ void PSWF_12D::buildGCF(float amp,
                                 float x0,
                                 float y0,
                                 float sigma_x,
                                 float sigma_y) {
  this->setKernelMemory();
  float x, y;
  float val;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      val = GCF(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, NULL);
      this->kernel[this->n * i + j] = val;
    }
  }
  this->copyKerneltoGPU();
};

__host__ void PSWF_12D::buildGCF() {
  this->setKernelMemory();
  float x, y;
  float val;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      val = GCF(this->amp, x, y, this->x0, this->y0, this->sigma_x,
                this->sigma_y, this->w, NULL);
      this->kernel[this->n * i + j] = val;
    }
  }
  this->copyKerneltoGPU();
};

__host__ float PSWF_12D::GCF(float amp,
                             float x,
                             float y,
                             float x0,
                             float y0,
                             float sigma_x,
                             float sigma_y,
                             float w,
                             float alpha) {
  float val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, w);
  return 1.0f / val;
};

__host__ CKernel* PSWF_12D::clone() const {
  return new PSWF_12D(*this);
};

__host__ void PSWF_12D::nameSelf() {
  this->name = "Prolate Spheroidal Wave Function (PSWF)";
};

namespace {
CKernel* CreateCKernel() {
  return new PSWF_12D;
}

const std::string name = "PSWF";
const bool RegisteredPSWF =
    registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};  // namespace
