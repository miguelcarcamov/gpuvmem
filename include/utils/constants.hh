#ifndef CONSTANTS_HH
#define CONSTANTS_HH

// Mathematical constants available to both CUDA and pure C++ code
// Using inline const (C++17) to allow multiple definitions in different translation units
#ifdef __CUDACC__
#include <math_constants.h>
inline const float PI = CUDART_PI_F;
inline const double PI_D = CUDART_PI;
#else
#include <cmath>
inline const float PI = 3.14159265358979323846f;
inline const double PI_D = 3.14159265358979323846;
#endif

#endif  // CONSTANTS_HH
