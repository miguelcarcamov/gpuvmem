#include "totalvariation.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

TVariation::TVariation(){
        this->name = "Total Variation";
};

float TVariation::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(totalvariation(p, device_S, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
}
void TVariation::calcGi(float *p, float *xi)
{
        DTVariation(p, device_DS, penalization_factor, mod, order, imageIndex);
};


void TVariation::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void TVariation::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, 0);
};

void TVariation::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateTVariation()
{
        return new TVariation;
}
const int TVariationId = 4;
const bool RegisteredTVariation = Singleton<FiFactory>::Instance().RegisterFi(TVariationId, CreateTVariation);
};
