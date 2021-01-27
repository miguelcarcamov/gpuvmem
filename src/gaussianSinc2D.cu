#include "gaussianSinc2D.cuh"

__host__ float gaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha)
{
        return amp*gaussian1D(1.0f, x, x0, sigma, w1, alpha)*sinc1D(1.0f, x, x0, sigma, w2);
};

__host__ float gaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha)
{
        float G = gaussian2D(1.0f, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        float S = sinc2D(1.0f, x, x0, y, y0, sigma_x, sigma_y, w2);
        return amp*G*S;
};

__host__ GaussianSinc2D::GaussianSinc2D() : CKernel(){
        this->w = 2.52;
        this->nameSelf();
};
__host__ GaussianSinc2D::GaussianSinc2D(int m, int n) : CKernel(m, n){
        this->w = 2.52;
        this->nameSelf();
};
__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float w, float w2) : CKernel(m, n, w){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, CKernel *gcf): CKernel(m, n, gcf){
        this->w = 2.52;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, Io *imageHandler) : CKernel(m, n, imageHandler){
        this->w = 2.52;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, Io *imageHandler, CKernel *gcf) : CKernel(m, n, imageHandler, gcf){
        this->w = 2.52;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float w, float w2, CKernel *gcf) : CKernel(m, n, w, gcf){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float w, float w2, Io *imageHandler, CKernel *gcf) : CKernel(m, n, w, imageHandler, gcf){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float dx, float dy, float w, float w2) : CKernel(m, n, dx, dy, w){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float dx, float dy, float w, float w2, CKernel *gcf) : CKernel(m, n, dx, dy, w, gcf){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float dx, float dy, float w, float w2, Io *imageHandler) : CKernel(m, n, dx, dy, w, imageHandler){
        this->w2 = w2;
        this->nameSelf();
};

__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float dx, float dy, float w, float w2, Io *imageHandler, CKernel *gcf) : CKernel(m, n, dx, dy, w, imageHandler, gcf){
        this->w2 = w2;
        this->nameSelf();
};

__host__ float GaussianSinc2D::getW2(){
        return this->w2;
};

__host__ float GaussianSinc2D::getAlpha(){
        return this->alpha;
};

__host__ void GaussianSinc2D::setW2(float w2){
        this->w2 = w2;
};

__host__ void GaussianSinc2D::setAlpha(float alpha){
        this->alpha = alpha;
};

__host__ void GaussianSinc2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussianSinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, this->w2, this->alpha);
                }
        }
        this->copyKerneltoGPU();

};

__host__ void GaussianSinc2D::buildKernel()
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussianSinc2D(this->amp, x, y, this->x0, this->y0, this->sigma_x, this->sigma_y, this->w, this->w2, this->alpha);
                }
        }
        this->copyKerneltoGPU();

};

__host__ CKernel* GaussianSinc2D::clone() const{
        return new GaussianSinc2D(*this);
};

__host__ void GaussianSinc2D::nameSelf(){
        this->name = "Gaussian Sinc";
};

namespace {
CKernel* CreateCKernel()
{
        return new GaussianSinc2D;
}

const std::string name = "GaussianSinc2D";
const bool RegisteredGaussianSinc2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
