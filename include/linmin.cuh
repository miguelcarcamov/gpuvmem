#ifndef LINMIN_CUH
#define LINMIN_CUH

#include "brent.cuh"
#include "f1dim.cuh"
#include "mnbrak.cuh"

__host__ void linmin(float* p, float* xi, float* fret, float (*func)(float*));
#endif
