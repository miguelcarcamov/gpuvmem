#ifndef BRENT_LINESEARCH_CUH
#define BRENT_LINESEARCH_CUH

#include "linesearch/linesearcher.cuh"
#include "linesearch/mnbrak.cuh"  // LineSearch1dFloatFunc

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

// Forward declaration of the Numerical Recipes brent function
__host__ float brent(float ax,
                     float bx,
                     float cx,
                     float tol,
                     float* xmin,
                     LineSearch1dFloatFunc f,
                     void* user);

#endif  // BRENT_LINESEARCH_CUH
