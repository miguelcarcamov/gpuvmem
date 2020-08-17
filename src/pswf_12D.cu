#include "pswf_12D.cuh"

__host__ __device__ void PSWF_12D::constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float x, y;
        float val;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1, this->m, this->n);
                        this->kernel[this->n * i + j] = val;
                }
        }

};

__device__ float PSWF_12D::constructGCF(float amp, float x0, float y0, float sigma_x, float sigma_y, float w, int M, int N)
{
        const int i = threadIdx.y + blockDim.y * blockIdx.y;
        const int j = threadIdx.x + blockDim.x * blockIdx.x;

        float x = (j - x0) * sigma_x;
        float y = (i - y0) * sigma_y;

        float val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, w, M, N);

        return 1.0f/val;

}


namespace {
CKernel* CreateCKernel()
{
        return new PSWF_12D;
}
const int CKERNELID = 6;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
