#include "chi2.cuh"
#include "imageProcessor.cuh"
#include <iostream>
#include <fstream>

extern long M, N;
extern int image_count;
extern int flag_opt;
extern float * penalizators;
extern int nPenalizators;

Chi2::Chi2()
{
        this->ip = new ImageProcessor();
};

void Chi2::configure(int penalizatorIndex, int imageIndex, int imageToAdd)
{
        this->imageIndex = imageIndex;
        this->order = order;
        this->mod = mod;
        this->ip->configure(image_count);

        if(penalizatorIndex != -1)
        {
                if(penalizatorIndex > (nPenalizators - 1) || penalizatorIndex < 0)
                {
                        printf("invalid index for penalizator (chi2)\n");
                        exit(-1);
                }else{
                        this->penalization_factor = penalizators[penalizatorIndex];
                }
        }

        gpuErrchk(cudaMalloc((void**)&result_dchi2, sizeof(float)*M*N*image_count));
        gpuErrchk(cudaMemset(result_dchi2, 0, sizeof(float)*M*N*image_count));
}

float Chi2::calcFi(float *p)
{
        float result = 0;
        result = penalization_factor * chi2(p, ip);
        return result;
};

void Chi2::calcGi(float *p, float *xi)
{
        dchi2(p, xi, result_dchi2, ip);
};

void Chi2::restartDGi()
{
        gpuErrchk(cudaMemset(result_dchi2, 0, sizeof(float)*M*N*image_count));
};

void Chi2::addToDphi(float *device_dphi)
{
        if(image_count == 1)
                linkAddToDPhi(device_dphi, result_dchi2, 0);
        if(image_count > 1) {
                gpuErrchk(cudaMemset(device_dphi, 0, sizeof(float)*M*N*image_count));
                gpuErrchk(cudaMemcpy(device_dphi, result_dchi2, sizeof(float)*N*M*image_count,cudaMemcpyDeviceToDevice));
        }
};

namespace {
Fi* CreateChi2()
{
        return new Chi2;
}
const int Chi2Id = 0;
const bool RegisteredChi2 = Singleton<FiFactory>::Instance().RegisterFi(Chi2Id, CreateChi2);
};
