#ifndef BB_MIN1_SEEDER_CUH
#define BB_MIN1_SEEDER_CUH

#include "linesearcher.cuh"

/**
 * @brief Barzilai-Borwein Adaptive Min1 step size seeder.
 * 
 * Uses the minimum of two Barzilai-Borwein step size estimates:
 * α_BB1 = (s_k^T s_k) / (s_k^T y_k)
 * α_BB2 = (s_k^T y_k) / (y_k^T y_k)
 * where s_k = x_{k+1} - x_k and y_k = g_{k+1} - g_k
 */
class BBMin1Seeder : public StepSizeSeeder {
 public:
  const char* methodName() const override { return "Barzilai-Borwein Min1"; }
  float seed(ObjectiveFunction* objective_function,
             float* current_point, float* search_direction,
             float* current_gradient, float prev_step_size = 1.0f,
             float* prev_gradient = nullptr,
             float* prev_point = nullptr) override;
};

#endif  // BB_MIN1_SEEDER_CUH
