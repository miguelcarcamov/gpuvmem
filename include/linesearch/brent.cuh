#ifndef BRENT_CUH
#define BRENT_CUH

#include "linesearch/linesearcher.cuh"

/**
 * @brief Brent's method for line search.
 * 
 * Uses Brent's method (combining golden section search and inverse parabolic interpolation)
 * to find the minimum. This is a robust method that combines the reliability of golden
 * section search with the speed of parabolic interpolation.
 */
class Brent : public LineSearcher {
 public:
  const char* methodName() const override { return "Brent's Method"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  // Brent's method parameters
  static constexpr int ITMAX = 500;
  static constexpr float CGOLD = 0.3819660f;
  static constexpr float ZEPS = 1.0e-10f;
};

// Forward declaration of the Numerical Recipes brent function
__host__ float brent(float ax,
                     float bx,
                     float cx,
                     float tol,
                     float* xmin,
                     float (*f)(float));

#endif  // BRENT_CUH
