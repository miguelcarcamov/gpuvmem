#ifndef LINESEARCH_UTILS_CUH
#define LINESEARCH_UTILS_CUH

#include "classes/objectivefunction.cuh"
#include "classes/image.cuh"

/**
 * @brief Evaluate the objective function along a line.
 * 
 * Computes f(x + α*d) where x is stored in device_pcom and d is stored in device_xicom.
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

#endif  // LINESEARCH_UTILS_CUH
