#include "laplacian.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

Laplacian::Laplacian(){
        this->name = "Laplacian";
};

float Laplacian::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(laplacian(p, device_S, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
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
