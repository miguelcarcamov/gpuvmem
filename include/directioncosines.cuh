#ifndef DIRECTIONCOSINES_CUH
#define DIRECTIONCOSINES_CUH
#include <ctgmath>

__host__ void direccos(double ra,
                       double dec,
                       double ra0,
                       double dec0,
                       double* l,
                       double* m);
#endif
