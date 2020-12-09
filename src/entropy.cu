#include "entropy.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

Entropy::Entropy(){
        this->name = "Entropy";
};

float Entropy::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(SEntropy(p, device_S, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
}
void Entropy::calcGi(float *p, float *xi)
{
        DEntropy(p, device_DS, penalization_factor, mod, order, imageIndex);
};


void Entropy::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void Entropy::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void Entropy::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateEntropy()
{
        return new Entropy;
}
const std::string name = "Entropy";
const bool RegisteredEntropy = registerCreationFunction<Fi, std::string>(name, CreateEntropy);
const bool RegisteredEntropyInt = registerCreationFunction<Fi, int>(0, CreateEntropy);
};
