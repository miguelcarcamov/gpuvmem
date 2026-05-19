#include "classes/optimizer.cuh"

#include "cli/optimization_reporting.hh"

#include <cmath>

__host__ Optimizer::~Optimizer() = default;

__host__ Optimizer::Optimizer()
    : of(nullptr),
      run_observer_(nullptr),
      optimization_sub_run_(0),
      optimization_sub_run_total_(0),
      image(nullptr),
      flag(0),
      total_iterations(100),
      current_iteration(0),
      ftol(1E-12f),
      gtol(1E-12f) {}

__host__ Optimizer::Optimizer(int total_iterations_in, float ftol_in)
    : of(nullptr),
      run_observer_(nullptr),
      optimization_sub_run_(0),
      optimization_sub_run_total_(0),
      image(nullptr),
      flag(0),
      total_iterations(total_iterations_in),
      current_iteration(0),
      ftol(ftol_in),
      gtol(1E-12f) {}

__host__ Optimizer::Optimizer(int total_iterations_in, float ftol_in, float gtol_in)
    : of(nullptr),
      run_observer_(nullptr),
      optimization_sub_run_(0),
      optimization_sub_run_total_(0),
      image(nullptr),
      flag(0),
      total_iterations(total_iterations_in),
      current_iteration(0),
      ftol(ftol_in),
      gtol(gtol_in) {}

__host__ float Optimizer::getFtol() { return ftol; }

__host__ float Optimizer::getGtol() { return gtol; }

__host__ int Optimizer::getCurrentIteration() { return current_iteration; }

__host__ int Optimizer::getTotalIterations() const { return total_iterations; }

__host__ void Optimizer::setRunObserver(gpuvmem::cli::IRunObserver* observer) {
  run_observer_ = observer;
}

__host__ gpuvmem::cli::IRunObserver* Optimizer::getRunObserver() const { return run_observer_; }

__host__ void Optimizer::setOptimizationSubRunContext(int sub_run, int sub_run_total,
                                                       const std::string& plane_label) {
  optimization_sub_run_ = sub_run;
  optimization_sub_run_total_ = sub_run_total;
  optimization_plane_label_ = plane_label;
}

__host__ void Optimizer::getOptimizationSubRunContext(int& sub_run, int& sub_run_total,
                                                       std::string& plane_label) const {
  sub_run = optimization_sub_run_;
  sub_run_total = optimization_sub_run_total_;
  plane_label = optimization_plane_label_;
}

__host__ void Optimizer::reportIteration(float phi, double wall_time_s) {
  gpuvmem::cli::notify_iteration(run_observer_, this, of, phi, wall_time_s);
}

__host__ void Optimizer::setImage(Image* image_in) { image = image_in; }

__host__ void Optimizer::setObjectiveFunction(ObjectiveFunction* of_in) { of = of_in; }

void Optimizer::setFlag(int flag_in) { flag = flag_in; }

void Optimizer::setFTol(float ftol_in) { ftol = ftol_in; }

void Optimizer::setGTol(float gtol_in) { gtol = gtol_in; }

void Optimizer::setTotalIterations(int iterations) { total_iterations = iterations; }

ObjectiveFunction* Optimizer::getObjectiveFunction() { return of; }

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
