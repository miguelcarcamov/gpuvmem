#ifndef LINESEARCH_UTILS_CUH
#define LINESEARCH_UTILS_CUH

#include "classes/objectivefunction.cuh"
#include "classes/image.cuh"

class Projection;

/**
 * @brief Evaluate the objective function along a line (delegates to current LineSearcher).
 *
 * Requires LineSearcher::ScopedSearchContext with a LineSearch1dEval bundle during search.
 *
 * @param alpha Step size along the search direction
 * @return Function value at the new point
 */
__host__ float evaluateLineFunction(float alpha);

/**
 * @brief Update a point along a search direction.
 * 
 * Computes p = p + α*d, applying constraints (e.g., positivity) as needed.
 * 
 * @param objective_function ObjectiveFunction instance (provides dimensions and CUDA config)
 * @param image Image instance (provides function mapping)
 * @param p Point to update (input/output)
 * @param d Search direction
 * @param alpha Step size
 */
__host__ void updatePoint(ObjectiveFunction* objective_function, Image* image,
                         float* p, float* d, float alpha);

/**
 * @brief Compute the directional derivative ∇f(x)^T*d.
 * 
 * @param gradient Gradient vector ∇f(x)
 * @param search_direction Search direction d
 * @return Dot product ∇f(x)^T*d
 */
__host__ float computeDirectionalDerivative(float* gradient, float* search_direction);

// Host wrappers for line search kernels (moved from functions.cu)
__host__ void defaultNewP(float* p, float* xi, float xmin, int image);
__host__ void defaultEvaluateXt(float* xt, float* pcom, float* xicom, float x, int image);
__host__ void particularNewP(float* p, float* xi, float xmin, int image);
__host__ void particularEvaluateXt(float* xt, float* pcom, float* xicom, float x, int image);

/** Extra projection pass on one image plane (Pyralysis `projection(parameter)` after trial). */
__host__ void applyProjectionToImagePlane(const Projection* proj, float* buffer, long N, long M,
                                          int image, unsigned blocks_x, unsigned blocks_y,
                                          unsigned threads_x, unsigned threads_y);

#endif  // LINESEARCH_UTILS_CUH
