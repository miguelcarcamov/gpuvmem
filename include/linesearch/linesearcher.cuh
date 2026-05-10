#ifndef LINESEARCHER_CUH
#define LINESEARCHER_CUH

#include "image.cuh"
#include "objectivefunction.cuh"
#include "optimization/projection.hh"
#include <string>
#include <utility>  // for std::pair
#include <memory>   // for std::unique_ptr

// Forward declarations
class Image;
class ObjectiveFunction;
struct LineSearch1dEval;

/**
 * @brief Base class for step size seeding strategies.
 * 
 * Step size seeders provide initial step size estimates for line search algorithms.
 * They can use gradient information, previous step sizes, or interpolation methods
 * to provide a good starting point for the line search.
 */
class StepSizeSeeder {
 public:
  virtual ~StepSizeSeeder() = default;

  /**
   * @brief Compute an initial step size estimate.
   * 
   * @param objective_function ObjectiveFunction instance (provides M, N, image_count, CUDA config)
   * @param current_point Current parameter point
   * @param search_direction Search direction vector (normalized descent direction)
   * @param current_gradient Current gradient vector
   * @param prev_step_size Previous step size (if available)
   * @param prev_gradient Previous gradient (if available, for BB methods)
   * @param prev_point Previous parameter point (if available, for BB methods)
   * @return Initial step size estimate
   */
  virtual float seed(ObjectiveFunction* objective_function,
                     float* current_point, float* search_direction,
                     float* current_gradient, float prev_step_size = 1.0f,
                     float* prev_gradient = nullptr,
                     float* prev_point = nullptr) = 0;

  /**
   * @brief Return the name of the seeding method.
   * 
   * @return Method name
   */
  virtual const char* methodName() const = 0;
};

// Forward declarations for seeder classes (defined in separate headers)
class BBMin1Seeder;
class BBMin2Seeder;
class BBAlternatingSeeder;
class CubicInterpolationSeeder;
class QuadraticInterpolationSeeder;

/**
 * @brief Base class for line search algorithms.
 * 
 * Line search algorithms find an appropriate step size along a search direction
 * to minimize the objective function. They can optionally use a StepSizeSeeder
 * to get an initial step size estimate.
 */
class LineSearcher {
 public:
  LineSearcher();
  virtual ~LineSearcher();

  /** Set the Image object (used by updatePoint and evaluateLineFunction). */
  void setImage(Image* image);

  /** @return Image object pointer */
  Image* getImage() const;

  /**
   * @brief Perform line search along the search direction.
   * 
   * Constraints (like positivity) are handled automatically through the
   * imageMap structure's newP function pointers.
   * 
   * The line searcher will automatically determine the initial step size:
   * - First iteration: uses initial_step_size_value (set via setInitialStepSize)
   * - Subsequent iterations: uses seeder (if available) or prev_step_size from history
   * 
   * @param current_point Current parameter point (input/output)
   * @param search_direction Search direction vector
   * @param objective_function Objective function to minimize
   * @param mask Optional mask for computations (can be nullptr)
   * @return Pair of (function_value, step_size)
   */
  virtual std::pair<float, float> search(
      float* current_point, float* search_direction,
      ObjectiveFunction* objective_function,
      float* mask = nullptr) = 0;

  /**
   * @brief Set the step size seeder to use.
   * 
   * @param seeder Step size seeder instance (ownership transferred)
   */
  void setStepSizeSeeder(std::unique_ptr<StepSizeSeeder> seeder);

  /** @return Pointer to seeder, or nullptr if not set */
  StepSizeSeeder* getStepSizeSeeder() const;

  /**
   * @brief Return the name of the line search method.
   * 
   * @return Method name
   */
  virtual const char* methodName() const = 0;

  virtual void setTolerance(float tol);
  virtual float getTolerance() const;

  /** Initial alpha when no history or seeder (default 1.0). */
  void setInitialStepSize(float initial_step_size);
  float getInitialStepSize() const;

  /**
   * @brief Update history after line search completes.
   * 
   * Called by optimizer after gradient is computed to update
   * history for the next iteration.
   * 
   * @param objective_function ObjectiveFunction instance (provides dimensions)
   * @param current_point Current parameter point (after update)
   * @param current_gradient Current gradient (after update)
   * @param step_size Step size that was used
   */
  void updateHistory(ObjectiveFunction* objective_function,
                    float* current_point, float* current_gradient, float step_size);
  
  /**
   * @brief Compute initial step size for next iteration using seeder.
   * 
   * Called by optimizer AFTER line search completes to compute initial alpha for NEXT iteration.
   * Uses the seeder with current and previous gradients/history.
   * The step size from the current iteration is stored in prev_step_size.
   * 
   * @param objective_function ObjectiveFunction instance (provides dimensions and CUDA config)
   * @param current_point Current parameter point (after line search update)
   * @param search_direction Search direction vector for next iteration
   * @param current_gradient Current gradient (just computed after line search)
   * @return Initial step size estimate for next iteration
   */
  float computeNextInitialAlpha(ObjectiveFunction* objective_function,
                               float* current_point, float* search_direction,
                               float* current_gradient);

  /**
   * @brief Evaluate the objective function along a line.
   *
   * Uses the active LineSearch1dEval from ScopedSearchContext (x, d on device in that bundle).
   *
   * @param alpha Step size along the search direction
   * @return Function value at the new point
   */
  float evaluateLineFunction(float alpha);

  /**
   * RAII: sets active LineSearcher (see lineSearchGetCurrent()), objective_function_, and optional 1D line-eval bundle
   * for evaluateLineFunction / mnbrak-style callbacks. Nest at the start of search().
   */
  class ScopedSearchContext {
   public:
    explicit ScopedSearchContext(LineSearcher* owner, ObjectiveFunction* of,
                                 const LineSearch1dEval* line_eval = nullptr);
    ~ScopedSearchContext();
    ScopedSearchContext(const ScopedSearchContext&) = delete;
    ScopedSearchContext& operator=(const ScopedSearchContext&) = delete;

   private:
    LineSearcher* owner_;
    LineSearcher* prev_current_;
    const LineSearch1dEval* prev_line_eval_;
  };

  void setObjectiveFunction(ObjectiveFunction* of);
  ObjectiveFunction* getObjectiveFunction() const;

  void setProjection(std::unique_ptr<Projection> projection);
  const Projection* getProjection() const;

  /** Move projection out (e.g. when replacing the line searcher). */
  std::unique_ptr<Projection> releaseProjection();

 protected:
  float tolerance;
  float initial_step_size_value;
  Image* image;
  ObjectiveFunction* objective_function_;
  const LineSearch1dEval* active_line_1d_eval_;
  std::unique_ptr<Projection> projection_;
  StepSizeSeeder* seeder_ptr;
  float* prev_point;
  float* prev_gradient;
  float prev_step_size;
  
  /**
   * @brief Compute initial step size using seeder or previous step size from history.
   * 
   * This helper method is called by concrete LineSearcher implementations
   * at the start of search() to get the initial step size. It first tries to use
   * the seeder (if available and history is ready), then falls back to previous
   * step size from history, and finally uses initial_step_size_value.
   * 
   * @param objective_function ObjectiveFunction instance (provides gradient via getCurrentGradient)
   * @param current_point Current parameter point
   * @param search_direction Search direction vector
   * @return Initial step size estimate
   */
  float computeInitialAlpha(ObjectiveFunction* objective_function,
                           float* current_point, float* search_direction);
};

/** Active LineSearcher while `LineSearcher::ScopedSearchContext` is alive (no extern). */
LineSearcher* lineSearchGetCurrent();

// Forward declarations for line searcher classes (defined in separate headers)
class GLLArmijo;
class BacktrackingArmijo;
class FistaBacktracking;
class FibonacciSearch;
class Fixed;
class GoldenSectionSearch;
class Brent;  // Already in linesearch/brent.cuh

#endif  // LINESEARCHER_CUH
