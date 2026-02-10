#ifndef MNBRAK_CUH
#define MNBRAK_CUH
__host__ void mnbrak(float* ax,
                     float* bx,
                     float* cx,
                     float* fa,
                     float* fb,
                     float* fc,
                     float (*func)(float));
#endif
