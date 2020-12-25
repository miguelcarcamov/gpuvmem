#ifndef L1NORM_CUH
#define L1NORM_CUH

#include "framework.cuh"
#include "functions.cuh"


class L1norm : public Fi
{
private:
float epsilon;
public:
L1norm();
L1norm(float epsilon);
float getEpsilon();
void setEpsilon(float epsilon);
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
};

#endif
