#ifndef BB_MIN2_SEEDER_CUH
#define BB_MIN2_SEEDER_CUH

#include "linesearcher.cuh"

/**
 * @brief Barzilai-Borwein Adaptive Min2 step size seeder.
 * 
 * Uses the maximum of two Barzilai-Borwein step size estimates.
 */
class BBMin2Seeder : public StepSizeSeeder {
 public:
  const char* methodName() const override { return "Barzilai-Borwein Min2"; }
  float seed(ObjectiveFunction* objective_function,
             float* current_point, float* search_direction,
             float* current_gradient, float prev_step_size = 1.0f,
             float* prev_gradient = nullptr,
             float* prev_point = nullptr) override;
};

#endif  // BB_MIN2_SEEDER_CUH
