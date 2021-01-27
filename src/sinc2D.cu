#include "sinc2D.cuh"

__host__ float sincf(float x)
{
        float value;
        if(x==0.0f)
                value = 1.0f;
        else
                value = sinf(PI*x)/(PI*x);

        return value;
};

__host__ float sinc1D(float amp, float x, float x0, float sigma, float w)
{
        float radius = distance(x, 0.0f, x0, 0.0f);
        float val = radius/(w*sigma);
        float s;
        if(radius < w*sigma)
                s = amp*sincf(val);
        else
                s = 0.0f;
        return s;
};

__host__ float sinc2D(float amp, float x, float x0, float y, float y0, float sigma_x, float sigma_y, float w)
{
        float s_x = sinc1D(1.0f, x, x0, sigma_x, w);
        float s_y = sinc1D(1.0f, y, y0, sigma_y, w);
        return amp*s_x*s_y;
};

__host__ Sinc2D::Sinc2D() : CKernel(){
        this->w =1.0f;
        this->nameSelf();
};
__host__  Sinc2D::Sinc2D(int m, int n) : CKernel(m, n){
        this->w =1.0f;
        this->nameSelf();
};
__host__ Sinc2D::Sinc2D(int m, int n, float w) : CKernel(m, n, w){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, CKernel *gcf): CKernel(m, n, gcf){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, Io *imageHandler) : CKernel(m, n, imageHandler){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, Io *imageHandler, CKernel *gcf) : CKernel(m, n, imageHandler, gcf){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy) : CKernel(m, n, dx, dy){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, CKernel *gcf) : CKernel(m, n, dx, dy, gcf){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, Io *imageHandler) : CKernel(m, n, dx, dy, imageHandler){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, Io *imageHandler, CKernel *gcf) : CKernel(m, n, dx, dy, imageHandler, gcf){
        this->w =1.0f;
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float w, CKernel *gcf) : CKernel(m, n, w, gcf){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float w, Io *imageHandler, CKernel *gcf) : CKernel(m, n, w, imageHandler, gcf){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, float w) : CKernel(m, n, dx, dy, w){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, float w, CKernel *gcf) : CKernel(m, n, dx, dy, w, gcf){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, float w, Io *imageHandler) : CKernel(m, n, dx, dy, w, imageHandler){
        this->nameSelf();
};

__host__ Sinc2D::Sinc2D(int m, int n, float dx, float dy, float w, Io *imageHandler, CKernel *gcf) : CKernel(m, n, dx, dy, w, imageHandler, gcf){
        this->nameSelf();
};

__host__ void Sinc2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = sinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w);
                }
        }
        this->copyKerneltoGPU();

};

__host__ void Sinc2D::buildKernel()
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = sinc2D(this->amp, x, y, this->x0, this->y0, this->sigma_x, this->sigma_y, this->w);
                }
        }
        this->copyKerneltoGPU();

};

__host__ float Sinc2D::GCF_fn(float amp, float nu, float w)
{
        if(fabs(nu) < w)
                return amp;
        else
                return 0.0f;
};

__host__ float Sinc2D::GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float distance_x = distance(x, y, x0, y0) * sigma_x;
        float distance_y = distance(x, y, x0, y0) * sigma_y;
        return GCF_fn(amp, distance_x, w*sigma_x)*GCF_fn(amp, distance_y, w*sigma_y);
};

__host__ CKernel* Sinc2D::clone() const{
        return new Sinc2D(*this);
};

__host__ void Sinc2D::nameSelf(){
        this->name = "Sinc";
};

namespace {
CKernel* CreateCKernel()
{
        return new Sinc2D;
}

const std::string name = "Sinc2D";
const bool RegisteredSinc2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
