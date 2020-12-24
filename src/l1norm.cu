#include "l1norm.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

L1norm::L1norm(){
        this->name = "L1 Norm";
};

L1norm::L1norm(float epsilon){
        this->name = "L1 Norm";
        this->epsilon = epsilon;
};

float L1norm::getEpsilon(){
        return this->epsilon;
};

float L1norm::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(L1Norm(p, device_S, penalization_factor, this->epsilon, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
}
void L1norm::calcGi(float *p, float *xi)
{
        DL1Norm(p, device_DS, penalization_factor, this->epsilon, mod, order, imageIndex);
};

void L1norm::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void L1norm::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void L1norm::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

void L1norm::setEpsilon(float epsilon)
{
        this->epsilon = epsilon;
};

namespace {
Fi* CreateL1norm()
{
        return new L1norm;
}
const std::string name = "L1-Norm";
const bool RegisteredL1norm = registerCreationFunction<Fi, std::string>(name, CreateL1norm);
};
