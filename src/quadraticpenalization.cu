#include "quadraticpenalization.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

QuadraticP::QuadraticP(){
};

float QuadraticP::calcFi(float *p)
{
        float result = 0.0;
        result = (penalization_factor)*( quadraticP(p, device_S, penalization_factor, mod, order, imageIndex) );
        return result;
}
void QuadraticP::calcGi(float *p, float *xi)
{
        DQuadraticP(p, device_DS, penalization_factor, mod, order, imageIndex);
};


void QuadraticP::restartDGi()
{
        gpuErrchk(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void QuadraticP::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void QuadraticP::configure(int penalizatorIndex, int imageIndex, int imageToAdd)
{
        this->imageIndex = imageIndex;
        this->order = order;
        this->mod = mod;
        this->imageToAdd = imageToAdd;

        if(imageIndex > image_count -1 || imageToAdd > image_count -1)
        {
                printf("There is no image for the provided index (QuadraticP)\n");
                exit(-1);
        }

        if(penalizatorIndex != -1)
        {
                if(penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0)
                {
                        printf("invalid index for penalizator (QuadraticP)\n");
                        exit(-1);
                }else{
                        this->penalization_factor = penalizators[penalizatorIndex];
                }
        }

        gpuErrchk(cudaMalloc((void**)&device_S, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_S, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_DS, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_DS, 0, sizeof(float)*M*N));

};

void QuadraticP::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateQuadraticP()
{
        return new QuadraticP;
}
const int QuadraticPId = 3;
const bool RegisteredQuadraticP = Singleton<FiFactory>::Instance().RegisterFi(QuadraticPId, CreateQuadraticP);
};
