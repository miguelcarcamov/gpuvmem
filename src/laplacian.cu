#include "laplacian.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

Laplacian::Laplacian(){
};

float Laplacian::calcFi(float *p)
{
        float result = 0.0;
        result = (penalization_factor)*( laplacian(p, device_S, penalization_factor, mod, order, imageIndex) );
        return result;
}
void Laplacian::calcGi(float *p, float *xi)
{
        DLaplacian(p, device_DS, penalization_factor, mod, order, imageIndex);
};


void Laplacian::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void Laplacian::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void Laplacian::configure(int penalizatorIndex, int imageIndex, int imageToAdd)
{
        this->imageIndex = imageIndex;
        this->order = order;
        this->mod = mod;
        this->imageToAdd = imageToAdd;

        if(imageIndex > image_count -1 || imageToAdd > image_count -1)
        {
                printf("There is no image for the provided index (Laplacian)\n");
                exit(-1);
        }

        if(penalizatorIndex != -1)
        {
                if(penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0)
                {
                        printf("invalid index for penalizator (laplacian)\n");
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

void Laplacian::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

namespace {
Fi* CreateLaplacian()
{
        return new Laplacian;
}
const int LaplacianId = 2;
const bool RegisteredLaplacian = Singleton<FiFactory>::Instance().RegisterFi(LaplacianId, CreateLaplacian);
};
