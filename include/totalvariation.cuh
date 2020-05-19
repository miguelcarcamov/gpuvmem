#ifndef TVVECTOR_CUH
#define TVVECTOR_CUH

#include "framework.cuh"
#include "functions.cuh"


class TVariation : public Fi
{
public:
TVariation();
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void configure(int penalizatorIndex, int imageIndex, int imageToAdd);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
};

#endif
