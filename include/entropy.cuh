#ifndef ENTROPY_CUH
#define ENTROPY_CUH

#include "framework.cuh"
#include "functions.cuh"


class Entropy : public Fi
{
private:
float prior_value;
public:
Entropy();
Entropy(float prior_value);
float getPrior();
void setPrior(float prior_value);
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
};

#endif
