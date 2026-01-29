#ifndef BACKTRACKING_ARMijo_LINESEARCH_CUH
#define BACKTRACKING_ARMijo_LINESEARCH_CUH

#include "linesearcher.cuh"

/**
 * @brief Backtracking Armijo line search.
 * 
 * Standard Armijo line search with backtracking.
 */
class BacktrackingArmijo : public LineSearcher {
 public:
  const char* methodName() const override { return "Backtracking Armijo"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  float rho = 0.5f;      // Backtracking factor
  float c = 1e-4f;       // Armijo constant
};

#endif  // BACKTRACKING_ARMijo_LINESEARCH_CUH
