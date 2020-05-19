#ifndef ERROR_CUH
#define ERROR_CUH

#include "framework.cuh"
#include "functions.cuh"

class SecondDerivateError : public Error
{
public:
SecondDerivateError(){
};
void calculateErrorImage(Image *I, Visibilities *v);
};

#endif
