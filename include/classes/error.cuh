#ifndef ERROR_CUH
#define ERROR_CUH

#include <image.cuh>
#include <visibilities.cuh>

class Error
{
public:
virtual void calculateErrorImage(Image *I, Visibilities *v) = 0;
};

#endif //ERROR_CUH
