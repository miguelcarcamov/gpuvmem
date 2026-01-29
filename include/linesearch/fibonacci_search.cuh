#ifndef FIBONACCI_SEARCH_LINESEARCH_CUH
#define FIBONACCI_SEARCH_LINESEARCH_CUH

#include "linesearcher.cuh"

/**
 * @brief Fibonacci search for line search.
 * 
 * Uses Fibonacci search method to find the minimum.
 */
class FibonacciSearch : public LineSearcher {
 public:
  const char* methodName() const override { return "Fibonacci Search"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  int max_iterations = 50;  // Maximum Fibonacci iterations
};

#endif  // FIBONACCI_SEARCH_LINESEARCH_CUH
