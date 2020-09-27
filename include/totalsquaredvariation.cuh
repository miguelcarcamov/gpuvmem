#ifndef TOTALSQUAREDVARIATION_CUH
#define TOTALSQUAREDVARIATION_CUH

#include "framework.cuh"
#include "functions.cuh"


class TotalSquaredVariationP : public Fi
{
public:
TotalSquaredVariationP();
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
};

#endif
