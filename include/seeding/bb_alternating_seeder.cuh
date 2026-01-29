#ifndef BB_ALTERNATING_SEEDER_CUH
#define BB_ALTERNATING_SEEDER_CUH

#include "linesearcher.cuh"

/**
 * @brief Barzilai-Borwein Alternating step size seeder.
 * 
 * Alternates between BB1 and BB2 estimates.
 */
class BBAlternatingSeeder : public StepSizeSeeder {
 public:
  const char* methodName() const override {
    return "Barzilai-Borwein Alternating";
  }
  float seed(ObjectiveFunction* objective_function,
             float* current_point, float* search_direction,
             float* current_gradient, float prev_step_size = 1.0f,
             float* prev_gradient = nullptr,
             float* prev_point = nullptr) override;

 private:
  bool use_bb1 = true;  // Alternate between BB1 and BB2
};

#endif  // BB_ALTERNATING_SEEDER_CUH
