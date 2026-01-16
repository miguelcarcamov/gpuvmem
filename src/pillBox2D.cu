#include "pillBox2D.cuh"

__host__ float pillBox1D(float amp, float x, float limit) {
  if (fabs(x) < limit)
    return amp;
  else
    return 0.0f;
};

__host__ float pillBox2D(float amp,
                         float x,
                         float y,
                         float limit_x,
                         float limit_y) {
  return pillBox1D(amp, x, limit_x) * pillBox1D(amp, y, limit_y);
};

/*
      Pill Box constructors and functions
*/

__host__ PillBox2D::PillBox2D() : CKernel() {
  this->nameSelf();
  this->setmn(1, 1);
};

__host__ PillBox2D::PillBox2D(int m, int n) : CKernel(m, n) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, float w) : CKernel(m, n, w) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, CKernel* gcf) : CKernel(m, n, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, Io* imageHandler)
    : CKernel(m, n, imageHandler) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, Io* imageHandler, CKernel* gcf)
    : CKernel(m, n, imageHandler, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, float dx, float dy)
    : CKernel(m, n, dx, dy) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, float dx, float dy, CKernel* gcf)
    : CKernel(m, n, dx, dy, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float dx,
                              float dy,
                              Io* imageHandler)
    : CKernel(m, n, dx, dy, imageHandler) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float dx,
                              float dy,
                              Io* imageHandler,
                              CKernel* gcf)
    : CKernel(m, n, dx, dy, imageHandler, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, float w, CKernel* gcf)
    : CKernel(m, n, w, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float w,
                              Io* imageHandler,
                              CKernel* gcf)
    : CKernel(m, n, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m, int n, float dx, float dy, float w)
    : CKernel(m, n, dx, dy, w) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float dx,
                              float dy,
                              float w,
                              CKernel* gcf)
    : CKernel(m, n, dx, dy, w, gcf) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float dx,
                              float dy,
                              float w,
                              Io* imageHandler)
    : CKernel(m, n, dx, dy, w, imageHandler) {
  this->nameSelf();
};

__host__ PillBox2D::PillBox2D(int m,
                              int n,
                              float dx,
                              float dy,
                              float w,
                              Io* imageHandler,
                              CKernel* gcf)
    : CKernel(m, n, dx, dy, w, imageHandler, gcf) {
  this->nameSelf();
};

__host__ void PillBox2D::buildKernel(float amp,
                                     float x0,
                                     float y0,
                                     float sigma_x,
                                     float sigma_y) {
  this->setKernelMemory();
  float limit_x = (this->m / 2.0f) * sigma_x;
  float limit_y = (this->n / 2.0f) * sigma_y;

  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      this->kernel[this->n * i + j] = pillBox2D(amp, x, y, limit_x, limit_y);
    }
  }
  // Normalize kernel (ensures proper gridding/degridding consistency)
  this->normalizeKernel();
  this->copyKerneltoGPU();
};

__host__ void PillBox2D::buildKernel() {
  this->setKernelMemory();
  float limit_x = (this->m / 2.0f) * this->sigma_x;
  float limit_y = (this->n / 2.0f) * this->sigma_y;

  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      this->kernel[this->n * i + j] =
          pillBox2D(this->amp, x, y, limit_x, limit_y);
    }
  }
  // Normalize kernel (ensures proper gridding/degridding consistency)
  this->normalizeKernel();
  this->copyKerneltoGPU();
};

__host__ void PillBox2D::buildGCF(float amp,
                                  float x0,
                                  float y0,
                                  float sigma_x,
                                  float sigma_y) {
  this->setKernelMemory();
  float limit_x = (this->m / 2.0f) * sigma_x;
  float limit_y = (this->n / 2.0f) * sigma_y;

  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * sigma_y;
      x = (j - this->support_x) * sigma_x;
      this->kernel[this->n * i + j] = GCF(amp, x, y, x0, y0, sigma_x, sigma_y);
    }
  }

  this->copyKerneltoGPU();
};

__host__ void PillBox2D::buildGCF() {
  this->setKernelMemory();
  float limit_x = (this->m / 2.0f) * this->sigma_x;
  float limit_y = (this->n / 2.0f) * this->sigma_y;

  float x, y;
  for (int i = 0; i < this->m; i++) {
    for (int j = 0; j < this->n; j++) {
      y = (i - this->support_y) * this->sigma_y;
      x = (j - this->support_x) * this->sigma_x;
      this->kernel[this->n * i + j] = GCF(this->amp, x, y, this->x0, this->y0,
                                          this->sigma_x, this->sigma_y);
    }
  }

  this->copyKerneltoGPU();
};

__host__ float PillBox2D::GCF(float amp,
                              float x,
                              float y,
                              float x0,
                              float y0,
                              float sigma_x,
                              float sigma_y) {
  return 1.0f;
};

__host__ CKernel* PillBox2D::clone() const {
  return new PillBox2D(*this);
};

__host__ void PillBox2D::nameSelf() {
  this->name = "Pill Box";
};

namespace {
CKernel* CreateCKernel() {
  return new PillBox2D;
}

const std::string name = "PillBox2D";
const bool RegisteredPillbox =
    registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};  // namespace
