#ifndef BRENT_LINESEARCH_CUH
#define BRENT_LINESEARCH_CUH

#include "linesearch/linesearcher.cuh"
#include "linesearch/mnbrak.cuh"  // LineSearch1dFloatFunc

/**
 * @brief Brent's method for line search.
 *
 * Pipeline mirrors Pyralysis ``Brent.search`` + ``setup_bracketing`` (fixed ``0, 1`` mnbrak
 * seed in ``golden_section/bracketing.py``): bracket with ``mnbrak``, then NR ``brent``.
 * A small default ``step_domain_floor`` matches Pyralysis ``GoldenSectionSearch`` /
 * ``FibonacciSearch`` use of ``domain_eps=1e-12`` in ``setup_bracketing`` (Brent in
 * Pyralysis omits ``domain_eps``; we apply the floor after minimization so the accepted
 * step stays forward and strictly positive when projection flattens ``φ`` near ``α=0``).
 */
class Brent : public LineSearcher {
 public:
  __host__ Brent();
  const char* methodName() const override { return "Brent"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;
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
