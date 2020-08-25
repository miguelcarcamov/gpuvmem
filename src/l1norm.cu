#include "l1norm.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

L1norm::L1norm(){
};

float L1norm::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(L1Norm(p, device_S, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
}
void L1norm::calcGi(float *p, float *xi)
{
        DL1Norm(p, device_DS, penalization_factor, mod, order, imageIndex);
};


void L1norm::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void L1norm::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void L1norm::configure(int penalizatorIndex, int imageIndex, int imageToAdd)
{
        this->imageIndex = imageIndex;
        this->order = order;
        this->mod = mod;
        this->imageToAdd = imageToAdd;

        if(imageIndex > image_count -1 || imageToAdd > image_count -1)
        {
                printf("There is no image for the provided index (L1 Norm)\n");
                exit(-1);
        }

        if(penalizatorIndex != -1)
        {
                if(penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0)
                {
                        printf("invalid index for penalizator (L1 Norm)\n");
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

void L1norm::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateL1norm()
{
        return new L1norm;
}
const int L1normId = 6;
const bool RegisteredL1norm = Singleton<FiFactory>::Instance().RegisterFi(L1normId, CreateL1norm);
};
