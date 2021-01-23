#include "pswf_12D.cuh"

__host__ void PSWF_12D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        float val;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1);
                        this->kernel[this->n * i + j] = val;
                }
        }
        this->copyKerneltoGPU();

};

__host__ float PSWF_12D::GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, w);
        return 1.0f/val;
}


namespace {
CKernel* CreateCKernel()
{
        return new PSWF_12D;
}

const std::string name = "PSWF";
const bool RegisteredPSWF = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
