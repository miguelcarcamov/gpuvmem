#ifndef GPUVMEM_LINESEARCH_GLOBALS_CUH
#define GPUVMEM_LINESEARCH_GLOBALS_CUH

// Global variables for linesearch implementations.
// These are used by Numerical Recipes routines (f1dim, etc.) to access
// current point and search direction data.
//
// Defined in src/linesearch/linesearch_globals.cu

extern float* device_pcom;      // Current point (device memory)
extern float* device_xicom;     // Search direction (device memory)
extern float (*nrfunc)(float*); // Function pointer (currently unused, set to nullptr)

#endif  // GPUVMEM_LINESEARCH_GLOBALS_CUH
