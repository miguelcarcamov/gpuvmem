#ifndef CUBIC_INTERPOLATION_SEEDER_CUH
#define CUBIC_INTERPOLATION_SEEDER_CUH

#include "linesearcher.cuh"

/**
 * @brief Cubic interpolation step size seeder.
 * 
 * Uses cubic interpolation based on function values and gradients.
 */
class CubicInterpolationSeeder : public StepSizeSeeder {
 public:
  const char* methodName() const override { return "Cubic Interpolation"; }
  float seed(ObjectiveFunction* objective_function,
             float* current_point, float* search_direction,
             float* current_gradient, float prev_step_size = 1.0f,
             float* prev_gradient = nullptr,
             float* prev_point = nullptr) override;
};

#endif  // CUBIC_INTERPOLATION_SEEDER_CUH
