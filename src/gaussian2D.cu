#include "gaussian2D.cuh"

__host__ float gaussian1D(float amp, float x, float x0, float sigma, float w,
                          float alpha) {
  float radius_x = distance(x, 0.0f, x0, 0.0f);
  float val = radius_x / (w * sigma);
  float val_alpha = powf(val, alpha);
  float G = amp * expf(-val_alpha);

  return G;
};

__host__ float gaussian2D(float amp, float x, float y, float x0, float y0,
                          float sigma_x, float sigma_y, float w, float alpha) {
  float radius_x = distance(x, 0.0f, x0, 0.0f);
  float radius_y = distance(0.0f, y, 0.0, y0);
  if (radius_x < w * sigma_x && radius_y < w * sigma_y) {
    float fx = radius_x / (w * sigma_x);
    float fy = radius_y / (w * sigma_y);

    float val_x = powf(fx, alpha);
    float val_y = powf(fy, alpha);
    float G = amp * expf(-1.0f * (val_x + val_y));
    return G;
  } else
    return 0.0f;
};

__host__ Gaussian2D::Gaussian2D() : CKernel() {
  this->w = 1.0f;
  this->nameSelf();
};
__host__ Gaussian2D::Gaussian2D(int m, int n) : CKernel(m, n) {
  this->w = 1.0f;
  this->nameSelf();
};
__host__ Gaussian2D::Gaussian2D(int m, int n, float w) : CKernel(m, n, w) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, CKernel *gcf)
    : CKernel(m, n, gcf) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, Io *imageHandler)
    : CKernel(m, n, imageHandler) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, Io *imageHandler, CKernel *gcf)
    : CKernel(m, n, imageHandler, gcf) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy)
    : CKernel(m, n, dx, dy) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy, CKernel *gcf)
    : CKernel(m, n, dx, dy, gcf) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy,
                                Io *imageHandler)
    : CKernel(m, n, dx, dy, imageHandler) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy,
                                Io *imageHandler, CKernel *gcf)
    : CKernel(m, n, dx, dy, imageHandler, gcf) {
  this->w = 1.0f;
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float w, CKernel *gcf)
    : CKernel(m, n, w, gcf) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float w, Io *imageHandler,
                                CKernel *gcf)
    : CKernel(m, n, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy, float w)
    : CKernel(m, n, dx, dy, w) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy, float w,
                                CKernel *gcf)
    : CKernel(m, n, dx, dy, w, gcf) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy, float w,
                                Io *imageHandler)
    : CKernel(m, n, dx, dy, w, imageHandler) {
  this->nameSelf();
};

__host__ Gaussian2D::Gaussian2D(int m, int n, float dx, float dy, float w,
                                Io *imageHandler, CKernel *gcf)
    : CKernel(m, n, dx, dy, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ float Gaussian2D::getAlpha() { return this->alpha; };

__host__ void Gaussian2D::setAlpha(float alpha) { this->alpha = alpha; };

__host__ void Gaussian2D::buildKernel(float amp, float x0, float y0,
                                      float sigma_x, float sigma_y) {
  this->setKernelMemory();
  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      this->kernel[this->n * i + j] =
          gaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, this->alpha);
    }
  }
  this->copyKerneltoGPU();
};

__host__ void Gaussian2D::buildKernel() {
  this->setKernelMemory();
  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      this->kernel[this->n * i + j] =
          gaussian2D(this->amp, x, y, this->x0, this->y0, this->sigma_x,
                     this->sigma_y, this->w, this->alpha);
    }
  }
  this->copyKerneltoGPU();
};

__host__ float Gaussian2D::GCF(float amp, float x, float y, float x0, float y0,
                               float sigma_x, float sigma_y, float w,
                               float alpha) {
  return gaussian2D(amp, PI * x, PI * y, PI * x0, PI * y0, sigma_x, sigma_y,
                    2.0f * w, alpha);
};

__host__ void Gaussian2D::buildGCF() {
  this->setKernelMemory();
  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      this->kernel[this->n * i + j] =
          GCF(this->amp, x, y, this->x0, this->y0, this->sigma_x, this->sigma_y,
              this->w, this->alpha);
    }
  }
  this->copyKerneltoGPU();
};

__host__ void Gaussian2D::buildGCF(float amp, float x0, float y0, float sigma_x,
                                   float sigma_y) {
  this->setKernelMemory();
  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      this->kernel[this->n * i + j] =
          GCF(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, this->alpha);
    }
  }
  this->copyKerneltoGPU();
};

__host__ CKernel *Gaussian2D::clone() const { return new Gaussian2D(*this); };

__host__ void Gaussian2D::nameSelf() { this->name = "Gaussian"; };

namespace {
CKernel *CreateCKernel() { return new Gaussian2D; }

const std::string name = "Gaussian2D";
const bool RegisteredGaussian2D =
    registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};  // namespace
