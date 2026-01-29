#ifndef BRENT_LINESEARCH_CUH
#define BRENT_LINESEARCH_CUH

#include "linesearcher.cuh"

/**
 * @brief Brent's method for line search.
 * 
 * Uses Brent's method (combination of golden section search and parabolic interpolation)
 * to find the minimum. This is the current implementation.
 */
class Brent : public LineSearcher {
 public:
  const char* methodName() const override { return "Brent"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  float tolerance = 1e-4f;  // Tolerance for convergence
};

#endif  // BRENT_LINESEARCH_CUH
