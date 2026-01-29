#ifndef GOLDEN_SECTION_SEARCH_LINESEARCH_CUH
#define GOLDEN_SECTION_SEARCH_LINESEARCH_CUH

#include "linesearcher.cuh"

/**
 * @brief Golden section search for line search.
 * 
 * Uses golden section search method to find the minimum.
 */
class GoldenSectionSearch : public LineSearcher {
 public:
  const char* methodName() const override { return "Golden Section Search"; }
  std::pair<float, float> search(float* current_point, float* search_direction,
                                  ObjectiveFunction* objective_function,
                                  float* mask = nullptr) override;

 protected:
  float golden_ratio = 0.618033988749895f;  // (sqrt(5) - 1) / 2
};

#endif  // GOLDEN_SECTION_SEARCH_LINESEARCH_CUH
