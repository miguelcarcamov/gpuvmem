#ifndef SECONDDERIVATEERROR_CUH
#define SECONDDERIVATEERROR_CUH

#include "error.cuh"

class Optimizer;

class SecondDerivateError : public Error {
 public:
  SecondDerivateError() : optimizer(nullptr), fg_scale(1.0f) {}
  void setOptimizer(Optimizer* o) { optimizer = o; }
  void setFgScale(float fg_scale) { this->fg_scale = fg_scale; }
  void calculateErrorImage(Image* I, Visibilities* v) override;

 private:
  Optimizer* optimizer;
  float fg_scale;
};

#endif  // SECONDDERIVATEERROR_CUH
