#include "totalsquaredvariation.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

TotalSquaredVariationP::TotalSquaredVariationP(){
  this->name = "Total Squared Variation";
};

float TotalSquaredVariationP::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(TotalSquaredVariation(p, device_S, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
}
void TotalSquaredVariationP::calcGi(float *p, float *xi)
{
        DTSVariation(p, device_DS, penalization_factor, mod, order, imageIndex);
};

void TotalSquaredVariationP::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void TotalSquaredVariationP::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void TotalSquaredVariationP::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateTotalSquaredVariationP()
{
        return new TotalSquaredVariationP;
}

const std::string name = "TotalSquaredVariation";
const bool RegisteredTotalSquaredVariation = registerCreationFunction<Fi, std::string>(name, CreateTotalSquaredVariation);
};
