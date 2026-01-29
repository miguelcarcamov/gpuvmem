#ifndef LBFGS_CUH
#define LBFGS_CUH

#include "linmin.cuh"
#include "classes/optimizer.cuh"
#include "functions.cuh"
#include "linesearcher.cuh"
#include "linesearch/brent.cuh"
#include <string>
#include <memory>

// Forward declaration to avoid circular dependency
class StepSizeSeeder;

/**
 * @brief Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.
 * 
 * This class implements the L-BFGS algorithm, a quasi-Newton optimization method
 * that approximates the inverse Hessian matrix using limited memory. The algorithm
 * is particularly suited for large-scale optimization problems where storing the
 * full Hessian matrix is computationally infeasible.
 * 
 * The implementation uses a two-loop recursion to efficiently compute search
 * directions without explicitly storing or forming the inverse Hessian matrix.
 * It maintains a circular buffer of correction pairs (s, y) and associated rho
 * scalars to approximate the Hessian inverse using limited historical information.
 */
class LBFGS : public Optimizer {
 public:
  __host__ LBFGS();
  
  __host__ int getK() override;
  __host__ void setK(int K) override;
  __host__ void allocateMemoryGpu() override;
  __host__ void deallocateMemoryGpu() override;
  __host__ void optimize() override;
  
  /**
   * @brief Set the line search algorithm to use.
   * 
   * @param searcher Line searcher instance (ownership transferred)
   * 
   * Note: To set a seeder, use: searcher->setStepSizeSeeder(std::make_unique<BBMin1Seeder>())
   * before calling setLineSearcher.
   */
  __host__ void setLineSearcher(std::unique_ptr<LineSearcher> searcher);

 protected:
  /**
   * @brief Compute the descent direction using two-loop recursion.
   * 
   * Implements the two-loop recursion to compute H_k^(-1) * g_k where H_k^(-1)
   * is approximated using limited memory BFGS updates. This avoids storing the
   * full inverse Hessian matrix.
   * 
   * @param gradient Current gradient vector g_k
   * @return Search direction vector H_k^(-1) * g_k (negative descent direction)
   */
  __host__ void computeDirection(float* gradient);

  /**
   * @brief Compute alpha coefficients for first loop of two-loop recursion.
   * 
   * Implements the iterative alpha computation as per standard L-BFGS algorithm.
   * 
   * @param gradient Current gradient vector g
   * @param par_M Number of correction pairs to use
   * @param lbfgs_it Current iteration index in circular buffer
   * @param alpha Output array for alpha coefficients
   */
  __host__ void computeAlphaCoefficients(float* gradient, int par_M, int lbfgs_it,
                                        float** alpha);

  /**
   * @brief Compute beta coefficients and update r in second loop.
   * 
   * Implements the iterative second loop as per standard L-BFGS algorithm.
   * 
   * @param r Intermediate vector r = gamma * q from first loop
   * @param alpha Alpha coefficients from first loop
   * @param par_M Number of correction pairs to use
   * @param lbfgs_it Current iteration index in circular buffer
   */
  __host__ void computeBetaCoefficients(float* r, float** alpha, int par_M,
                                        int lbfgs_it);

  /**
   * @brief Compute gamma scaling factor for Hessian approximation.
   * 
   * Computes the BFGS scaling factor: gamma = (s^T y) / (y^T y)
   * 
   * @param par_M Number of correction pairs to use
   * @param lbfgs_it Current iteration index in circular buffer
   * @return Gamma scaling factor
   */
  __host__ float computeScalingFactor(int par_M, int lbfgs_it);

  /**
   * @brief Update LBFGS iteration history.
   * 
   * Updates the correction pairs where:
   * s_k = x_{k+1} - x_k and y_k = g_{k+1} - g_k
   * 
   * @param iteration Current iteration number
   */
  __host__ void updateHistory(int iteration);

  /**
   * @brief Perform a single optimization iteration.
   * 
   * @param iteration Current iteration number
   * @param prev_function_value Previous function value
   * @return New function value after iteration
   */
  __host__ float performIteration(int iteration, float prev_function_value);

  /**
   * @brief Check for function convergence.
   * 
   * @param new_value New function value
   * @param prev_value Previous function value
   * @return true if converged, false otherwise
   */
  __host__ bool checkFunctionConvergence(float new_value, float prev_value);

  /**
   * @brief Check for gradient convergence.
   * 
   * @return true if converged, false otherwise
   */
  __host__ bool checkGradientConvergence();

  /**
   * @brief Initialize optimization state.
   * 
   * @return Initial function value
   */
  __host__ float initializeOptimizationState();

  // Memory pointers for CUDA operations
  float* d_s;           // Correction pairs: s_k = x_{k+1} - x_k
  float* d_y;           // Correction pairs: y_k = g_{k+1} - g_k
  float* xi;            // Current gradient / search direction
  float* xi_old;        // Previous gradient
  float* p_old;         // Previous parameter values
  float* norm_vector;   // Temporary storage for gradient norm computation
  float* d_r;           // Temporary storage for second loop
  float* d_q;           // Temporary storage for first loop
  float* aux_vector;    // Temporary storage for dot products

  // Optimization state
  float fret = 0.0f;    // Function value after line search
  float fp = 0.0f;      // Previous function value
  float max_per_it = 0.0f;  // Maximum gradient component
  int configured = 1;   // Configuration flag (1 = needs configuration)
  int K = 100;          // Maximum number of correction pairs (memory limit)
  
  // Line search (opaque pointer to avoid circular dependency in header)
  void* linesearcher_ptr;  // Line search algorithm (LineSearcher*)
                          // Note: LineSearcher owns its own seeder internally
  
  // Previous step size (used as fallback initial_alpha for line search)
  float prev_step_size;   // Previous step size

 private:
  /**
   * @brief Map logical index k to circular buffer index.
   * 
   * @param k Logical index (0 = oldest, par_M-1 = newest)
   * @param par_M Number of correction pairs
   * @param lbfgs_it Current iteration index in circular buffer
   * @return Circular buffer index
   */
  __host__ int mapToCircularBuffer(int k, int par_M, int lbfgs_it);
};

#endif  // LBFGS_CUH
