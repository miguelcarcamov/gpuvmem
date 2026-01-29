#ifndef GLL_ARMijo_LINESEARCH_CUH
#define GLL_ARMijo_LINESEARCH_CUH

#include "linesearch/backtracking_armijo.cuh"

/**
 * @brief GLL (Grippo-Lampariello-Lucidi) Armijo line search.
 * 
 * Implements the GLL nonmonotone Armijo line search with backtracking.
 * Inherits from BacktrackingArmijo and extends it with non-monotonicity.
 */
class GLLArmijo : public BacktrackingArmijo {
 public:
  const char* methodName() const override { return "GLL Armijo"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  int M = 10;           // Nonmonotone parameter (memory window)
};

#endif  // GLL_ARMijo_LINESEARCH_CUH
