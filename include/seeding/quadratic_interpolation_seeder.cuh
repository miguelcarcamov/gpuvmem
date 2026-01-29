#ifndef QUADRATIC_INTERPOLATION_SEEDER_CUH
#define QUADRATIC_INTERPOLATION_SEEDER_CUH

#include "linesearcher.cuh"

/**
 * @brief Quadratic interpolation step size seeder.
 * 
 * Uses quadratic interpolation based on function values.
 */
class QuadraticInterpolationSeeder : public StepSizeSeeder {
 public:
  const char* methodName() const override { return "Quadratic Interpolation"; }
  float seed(ObjectiveFunction* objective_function,
             float* current_point, float* search_direction,
             float* current_gradient, float prev_step_size = 1.0f,
             float* prev_gradient = nullptr,
             float* prev_point = nullptr) override;
};

#endif  // QUADRATIC_INTERPOLATION_SEEDER_CUH
