#ifndef SECOND_DERIVATE_ERROR_CUH
#define SECOND_DERIVATE_ERROR_CUH

#include "framework.cuh"

class SecondDerivateError : public Error {
 public:
  SecondDerivateError() : fg_scale(1.0f), optimizer(NULL) {};
  void calculateErrorImage(Image* I, Visibilities* v);
  void setFgScale(float fg_scale) { this->fg_scale = fg_scale; }
  void setOptimizer(Optimizer* opt) { this->optimizer = opt; }

 private:
  float fg_scale;
  Optimizer* optimizer;
};

#endif
