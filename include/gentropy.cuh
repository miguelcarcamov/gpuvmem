#ifndef GENTROPY_CUH
#define GENTROPY_CUH

#include "framework.cuh"
#include "functions.cuh"


class GEntropy : public Fi
{
private:
float *prior;
float normalization_factor;
public:
GEntropy();
GEntropy(float *prior);
GEntropy(float *prior, float normalization_factor);
GEntropy(std::vector<float> prior);
GEntropy(std::vector<float> prior, float normalization_factor);
~GEntropy();
float getNormalizationFactor();
void setNormalizationFactor(float normalization_factor);
void setPrior(float *prior);
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
void normalizePrior();
};

#endif
