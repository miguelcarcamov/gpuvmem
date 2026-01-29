#ifndef CONJUGATEGRADIENT_CUH
#define CONJUGATEGRADIENT_CUH

#include "linmin.cuh"
#include "classes/optimizer.cuh"
#include "functions.cuh"
#include <string>
#include <memory>

// Forward declaration to avoid circular dependency
class LineSearcher;
class StepSizeSeeder;

/**
 * @brief Exception class for zero gradient norm errors.
 * 
 * This exception is raised when the gradient norm becomes zero during optimization,
 * indicating either convergence to a critical point or a numerical issue.
 */
class GradientNormError : public std::exception {
 public:
  const char* what() const noexcept override {
    return "Gradient norm is zero - cannot compute conjugate gradient parameter";
  }
};

/**
 * @brief Abstract base class for Conjugate Gradient optimization methods.
 * 
 * The conjugate gradient method is an iterative optimization algorithm that uses
 * gradient information to find the minimum of a differentiable function. At each
 * iteration, the search direction is computed as a combination of the negative
 * gradient and a scaled version of the previous search direction.
 * 
 * The general update rule for the search direction is:
 *   d_k = -g_k + β_k * d_{k-1}
 * 
 * where g_k is the gradient at iteration k and β_k is the conjugate gradient
 * parameter that determines the specific variant of the method.
 */
class ConjugateGradient : public Optimizer {
 public:
  __host__ ConjugateGradient();
  
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
   * 
   * Note: To set initial step size, use: searcher->setInitialStepSize(value)
   * before calling setLineSearcher.
   */
  __host__ void setLineSearcher(std::unique_ptr<LineSearcher> searcher);

 protected:
  /**
   * @brief Compute the conjugate gradient parameter β_k.
   * 
   * This method calculates the parameter used to update the search direction.
   * The specific formula depends on the chosen variant (implemented in subclasses).
   * 
   * @param grad Current gradient g_{k+1}
   * @param grad_prev Previous gradient g_k
   * @param dir_prev Previous search direction d_k
   * @return The conjugate gradient parameter β_k
   * @throws GradientNormError if previous gradient norm is zero
   */
  __host__ float conjugateGradientParameter(float* grad, float* grad_prev, float* dir_prev);

  /**
   * @brief Perform a single optimization iteration.
   * 
   * @param iteration Current iteration number
   * @param prev_function_value Previous function value
   * @param prev_gradient Previous gradient
   * @param prev_search_direction Previous search direction
   * @return New function value after iteration
   */
  __host__ float performIteration(int iteration, float prev_function_value,
                                   float* prev_gradient, float* prev_search_direction);

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
   * @param current_gradient Current gradient
   * @param function_value Current function value
   * @return true if converged, false otherwise
   */
  __host__ bool checkGradientConvergence(float* current_gradient, float function_value);

  /**
   * @brief Initialize optimization state.
   * 
   * @param prev_gradient Output: initial gradient
   * @param prev_search_direction Output: initial search direction
   * @return Initial function value
   */
  __host__ float initializeOptimizationState(float*& prev_gradient, float*& prev_search_direction);

  /**
   * @brief Return the name of the specific conjugate gradient method variant.
   * 
   * Each subclass implements a specific variant and should return a descriptive name.
   * 
   * @return The human-readable method variant name
   */
  virtual const char* methodName() const = 0;

  /**
   * @brief Compute the specific conjugate gradient parameter β_k.
   * 
   * This method implements the specific formula for computing β_k according to
   * the chosen conjugate gradient variant.
   * 
   * @param grad Current gradient g_{k+1}
   * @param grad_prev Previous gradient g_k
   * @param dir_prev Previous search direction d_k
   * @param norm2_grad_prev Precomputed squared norm of previous gradient
   * @return The computed parameter β_k
   */
  virtual float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                                  float* dir_prev,
                                                  float norm2_grad_prev) = 0;

  // Memory pointers for CUDA operations
  float* device_g;        // Previous gradient (g_k)
  float* device_h;        // Previous search direction (d_k)
  float* xi;              // Current gradient (g_{k+1}) / search direction
  float* temp;             // Temporary storage for gradient convergence check
  float* device_gg_vector; // Temporary storage for dot product reductions
  float* device_dgg_vector; // Temporary storage for dot product reductions

  // Optimization state
  float fret = 0.0f;      // Function value after line search
  float fp = 0.0f;        // Previous function value
  int configured = 1;     // Configuration flag (1 = needs configuration)
  
  // Line search (opaque pointer to avoid circular dependency in header)
  void* linesearcher_ptr;  // Line search algorithm (LineSearcher*)
                          // Note: LineSearcher owns its own seeder internally
  
  // Previous step size (used as fallback initial_alpha for line search)
  float prev_step_size;   // Previous step size
};

/**
 * @brief Hestenes-Stiefel conjugate gradient method variant.
 * 
 * The Hestenes-Stiefel method provides the theoretical foundation for most
 * conjugate gradient variants.
 * 
 * β_k^{HS} = (g_{k+1}^T (g_{k+1} - g_k)) / ((g_{k+1} - g_k)^T d_k)
 */
class HestenesStiefel : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Hestenes-Stiefel"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief Fletcher-Reeves conjugate gradient method variant.
 * 
 * The Fletcher-Reeves method ensures global convergence under mild conditions.
 * 
 * β_k^{FR} = ||g_{k+1}||^2 / ||g_k||^2
 */
class FletcherReeves : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Fletcher-Reeves"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief Polak-Ribière-Polyak conjugate gradient method variant.
 * 
 * Also known as the PRP method, this variant often exhibits superior performance
 * compared to Fletcher-Reeves, especially for nonlinear optimization problems.
 * 
 * β_k^{PRP} = (g_{k+1}^T (g_{k+1} - g_k)) / ||g_k||^2
 */
class PolakRibiere : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Polak-Ribiere-Polyak"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief Liu-Storey conjugate gradient method variant.
 * 
 * The Liu-Storey method has shown good performance for image restoration and
 * unconstrained optimization problems.
 * 
 * β_k^{LS} = -(y_k^T g_{k+1}) / (g_k^T d_k)
 * where y_k = g_{k+1} - g_k
 */
class LiuStorey : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Liu-Storey"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief Dai-Yuan conjugate gradient method variant.
 * 
 * The Dai-Yuan method is designed to ensure global convergence without requiring
 * exact line searches.
 * 
 * β_k^{DY} = ||g_{k+1}||^2 / ((g_{k+1} - g_k)^T d_k)
 */
class DaiYuan : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Dai-Yuan"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief Hager-Zhang conjugate gradient method variant.
 * 
 * The Hager-Zhang method is designed to achieve better practical performance
 * while maintaining theoretical convergence guarantees.
 * 
 * β_k^{HZ} = (y_k^T g_{k+1} / (d_k^T y_k)) - 2(||y_k||^2 (d_k^T g_{k+1}) / (d_k^T y_k)^2)
 * where y_k = g_{k+1} - g_k
 */
class HagerZhang : public ConjugateGradient {
 public:
  const char* methodName() const override { return "Hager-Zhang"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

/**
 * @brief RMIL (Rivaie-Mamat-Ismail-Abashar) conjugate gradient method variant.
 * 
 * The RMIL method has shown good performance for unconstrained optimization problems,
 * particularly in image recovery and robotic modeling applications.
 * 
 * β_k^{RMIL} = (g_{k+1}^T y_k) / (d_k^T (d_k - g_k))
 * where y_k = g_{k+1} - g_k
 */
class RMIL : public ConjugateGradient {
 public:
  const char* methodName() const override { return "RMIL"; }

 protected:
  float computeConjugateGradientParameter(float* grad, float* grad_prev,
                                          float* dir_prev,
                                          float norm2_grad_prev) override;
};

// CUDA kernel declarations for computing dot products
__global__ void computeDotProduct(float* result, float* vec1, float* vec2,
                                  long N, long M, int image);

__global__ void computeGradientDifference(float* grad_diff, float* grad,
                                          float* grad_prev, long N, long M,
                                          int image);

__global__ void computeNorm2Gradient(float* result, float* grad, long N, long M,
                                    int image);

#endif  // CONJUGATEGRADIENT_CUH
