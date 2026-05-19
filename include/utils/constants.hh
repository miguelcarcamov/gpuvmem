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

// Angle conversion constants
inline const float RPDEG = (PI / 180.0f);
inline const double RPDEG_D = (PI_D / 180.0);
inline const float RPARCSEC = (PI / (180.0f * 3600.0f));  // Radians per arcsecond
inline const double RPARCSEC_D = (PI_D / (180.0 * 3600.0));
inline const float RPARCM = (PI / (180.0f * 60.0f));  // Radians per arcminute
inline const double RPARCM_D = (PI_D / (180.0 * 60.0));

// Primary beam constant (first null of Airy disk)
inline const float RZ = 1.2196698912665045f;

#endif  // CONSTANTS_HH
