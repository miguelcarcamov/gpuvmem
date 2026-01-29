#ifndef FISTA_BACKTRACKING_LINESEARCH_CUH
#define FISTA_BACKTRACKING_LINESEARCH_CUH

#include "linesearch/backtracking_armijo.cuh"

/**
 * @brief FISTA backtracking line search.
 * 
 * Backtracking line search used in FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
 * Inherits from BacktrackingArmijo and extends it with FISTA-specific backtracking logic.
 */
class FistaBacktracking : public BacktrackingArmijo {
 public:
  const char* methodName() const override { return "FISTA Backtracking"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  float initial_lipschitz = 1.0f;  // Initial Lipschitz constant
  float prev_lipschitz = -1.0f;    // Previous Lipschitz constant (for warm start)
};

#endif  // FISTA_BACKTRACKING_LINESEARCH_CUH
