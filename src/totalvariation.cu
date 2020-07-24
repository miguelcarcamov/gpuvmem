#include "totalvariation.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

TVariation::TVariation(){
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

void TVariation::configure(int penalizatorIndex, int imageIndex, int imageToAdd)
{
        this->imageIndex = imageIndex;
        this->order = order;
        this->mod = mod;
        this->imageToAdd = imageToAdd;

        if(imageIndex > image_count -1 || imageToAdd > image_count -1)
        {
                printf("There is no image for the provided index (TVariation)\n");
                exit(-1);
        }

        if(penalizatorIndex != -1)
        {
                if(penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0)
                {
                        printf("invalid index for penalizator (TVariation)\n");
                        exit(-1);
                }else{
                        this->penalization_factor = penalizators[penalizatorIndex];
                }
        }

        checkCudaErrors(cudaMalloc((void**)&device_S, sizeof(float)*M*N));
        checkCudaErrors(cudaMemset(device_S, 0, sizeof(float)*M*N));

        checkCudaErrors(cudaMalloc((void**)&device_DS, sizeof(float)*M*N));
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
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
