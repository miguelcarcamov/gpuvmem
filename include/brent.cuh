#ifndef BRENT_CUH
#define BRENT_CUH
#include <stdio.h>
__host__ float brent(float ax,
                     float bx,
                     float cx,
                     float tol,
                     float* xmin,
                     float (*f)(float));
#endif
