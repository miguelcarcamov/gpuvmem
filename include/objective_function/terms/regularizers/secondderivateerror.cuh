#ifndef SECOND_DERIVATE_ERROR_CUH
#define SECOND_DERIVATE_ERROR_CUH

#include "error.cuh"

class Optimizer;

class SecondDerivateError : public Error {
 public:
  SecondDerivateError() : fg_scale(1.0f), optimizer(nullptr) {}
  void calculateErrorImage(Image* I, Visibilities* v) override;
  void setFgScale(float fg_scale) { this->fg_scale = fg_scale; }
  void setOptimizer(Optimizer* opt) { this->optimizer = opt; }

 private:
  float fg_scale;
  Optimizer* optimizer;
};

#endif
