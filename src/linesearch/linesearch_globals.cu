/* Global variables for linesearch implementations.
 * These are used by Numerical Recipes routines (f1dim, etc.) to access
 * current point and search direction data.
 */

#include "linesearch/linesearch_globals.cuh"

// Global variables for f1dim and Numerical Recipes line search routines
float* device_pcom = nullptr;      // Current point (device memory)
float* device_xicom = nullptr;     // Search direction (device memory)
float (*nrfunc)(float*) = nullptr; // Function pointer (currently unused, set to nullptr)
