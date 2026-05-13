#include "classes/optimizer.cuh"

#include <cmath>

__host__ Optimizer::~Optimizer() = default;

__host__ Optimizer::Optimizer()
    : of(nullptr),
      image(nullptr),
      flag(0),
      total_iterations(100),
      current_iteration(0),
      ftol(1E-12f),
      gtol(1E-12f) {}

__host__ Optimizer::Optimizer(int total_iterations_in, float ftol_in)
    : of(nullptr),
      image(nullptr),
      flag(0),
      total_iterations(total_iterations_in),
      current_iteration(0),
      ftol(ftol_in),
      gtol(1E-12f) {}

__host__ Optimizer::Optimizer(int total_iterations_in, float ftol_in, float gtol_in)
    : of(nullptr),
      image(nullptr),
      flag(0),
      total_iterations(total_iterations_in),
      current_iteration(0),
      ftol(ftol_in),
      gtol(gtol_in) {}

__host__ float Optimizer::getFtol() { return ftol; }

__host__ float Optimizer::getGtol() { return gtol; }

__host__ int Optimizer::getCurrentIteration() { return current_iteration; }

__host__ void Optimizer::setImage(Image* image_in) { image = image_in; }

__host__ void Optimizer::setObjectiveFunction(ObjectiveFunction* of_in) { of = of_in; }

void Optimizer::setFlag(int flag_in) { flag = flag_in; }

void Optimizer::setFTol(float ftol_in) { ftol = ftol_in; }

void Optimizer::setGTol(float gtol_in) { gtol = gtol_in; }

void Optimizer::setTotalIterations(int iterations) { total_iterations = iterations; }

ObjectiveFunction* Optimizer::getObjectiveFunction() { return of; }

__host__ int Optimizer::getK() { return 0; }

__host__ void Optimizer::setK(int /*K*/) {}

__host__ void Optimizer::setLineSearcher(std::unique_ptr<LineSearcher> /*searcher*/) {}

__host__ void Optimizer::setProjection(std::unique_ptr<Projection> /*projection*/) {}

__host__ bool Optimizer::objectiveSequenceWithinTolerance(float f_new,
                                                          float f_prev) const {
  if (!std::isfinite(f_new) || !std::isfinite(f_prev)) return false;
  const float diff = fabsf(f_new - f_prev);
  if (!(diff > 0.0f)) return true;
  const float scale = 1.0f + std::fmax(fabsf(f_new), fabsf(f_prev));
  return diff <= this->ftol * scale;
}
