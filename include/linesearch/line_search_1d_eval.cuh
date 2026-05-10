#ifndef LINE_SEARCH_1D_EVAL_CUH
#define LINE_SEARCH_1D_EVAL_CUH

class Image;
class ObjectiveFunction;

/**
 * @brief Explicit state for the 1D objective along a line, f(α) = F(x + α d).
 *
 * Mirrors Pyralysis `f1dim(...)` returning a callable: all data needed to evaluate
 * the reduced objective is carried here instead of file-scope globals.
 */
struct LineSearch1dEval {
  float* device_pcom = nullptr;       // Current point x (device)
  float* device_xicom = nullptr;      // Search direction d (device)
  Image* image = nullptr;
  ObjectiveFunction* objective_function = nullptr;
};

__host__ float lineSearch1dEval(const LineSearch1dEval* ctx, float x);

/** Numerical Recipes-style callback: @a user points to a LineSearch1dEval. */
__host__ float lineSearch1dEvalThunk(float x, void* user);

#endif  // LINE_SEARCH_1D_EVAL_CUH
