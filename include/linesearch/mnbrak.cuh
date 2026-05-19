#ifndef MNBRAK_CUH
#define MNBRAK_CUH

/** 1D objective f(x) with explicit @a user (Pyralysis-style callable, no globals). */
using LineSearch1dFloatFunc = float (*)(float x, void* user);

__host__ void mnbrak(float* ax,
                     float* bx,
                     float* cx,
                     float* fa,
                     float* fb,
                     float* fc,
                     LineSearch1dFloatFunc func,
                     void* user);
#endif
