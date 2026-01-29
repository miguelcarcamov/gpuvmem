#ifndef FIXED_LINESEARCH_CUH
#define FIXED_LINESEARCH_CUH

#include "linesearcher.cuh"

/**
 * @brief Fixed step size line search.
 * 
 * Uses a fixed step size (useful for testing or when step size is known).
 */
class Fixed : public LineSearcher {
 public:
  const char* methodName() const override { return "Fixed"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

  /**
   * @brief Set the fixed step size.
   * 
   * @param step_size Fixed step size to use
   */
  void setStepSize(float step_size) { fixed_step_size = step_size; }

  /**
   * @brief Get the fixed step size.
   * 
   * @return Fixed step size
   */
  float getStepSize() const { return fixed_step_size; }

 protected:
  float fixed_step_size = 1.0f;
};

#endif  // FIXED_LINESEARCH_CUH
