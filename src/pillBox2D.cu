#include "pillBox2D.cuh"

__host__ void PillBox2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float limit_x = this->m * sigma_x / 2;
        float limit_y = this->n * sigma_y / 2;

        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = pillBox2D(amp, x, y, limit_x, limit_y);
                }
        }

        this->copyKerneltoGPU();
};

__host__ float PillBox2D::GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        return 1.0f;
}

namespace {
CKernel* CreateCKernel()
{
        return new PillBox2D;
}

const std::string name = "PillBox2D";
const bool RegisteredPillbox = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
