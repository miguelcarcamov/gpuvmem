#ifndef SECOND_DERIVATE_ERROR_CUH
#define SECOND_DERIVATE_ERROR_CUH

#include "framework.cuh"

class SecondDerivateError : public Error
{
public:
SecondDerivateError(){
};
void calculateErrorImage(Image *I, Visibilities *v);
};

#endif
