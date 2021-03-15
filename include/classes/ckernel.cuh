#ifndef CKERNEL_CUH
#define CKERNEL_CUH

#include "io.cuh"

class CKernel
{
public:
__host__ virtual void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y) = 0;
__host__ virtual void buildKernel() = 0;
__host__ virtual float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y){return;};
__host__ virtual float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w){return;};
__host__ virtual float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha){return;};
__host__ virtual void buildGCF(float amp, float x0, float y0, float sigma_x, float sigma_y) {};
__host__ virtual void buildGCF() {};
__host__ virtual CKernel * getGCF(){return this->gcf;};
__host__ virtual void fillGCFvalues(float amp, float x0, float y0, float sigma_x, float sigma_y){this->gcf->buildGCF(amp, x0, y0, sigma_x, sigma_y);};
__host__ virtual void createMemberGCF(){};
__host__ virtual void setGCF(CKernel * gcf){this->gcf = gcf;};
__host__ virtual float * getGCFGPU(){return this->gcf->getGPUKernel();};
__host__ virtual std::vector<float> getGCFCPU(){return this->gcf->getKernel();};
__host__ virtual float* getGCFCPUPointer(){return this->gcf->getKernelPointer();};
__host__ virtual void printGCFCPU(){return this->gcf->printCKernel();};
__host__ virtual void printGCFGPU(){return this->gcf->printGPUCKernel();};

// Virtual functions for Gaussian and GaussianSinc
__host__ virtual float getAlpha(){return;};
__host__ virtual void setAlpha(float alpha){};

__host__ virtual float getW2(){return;};
__host__ virtual void setW2(float w2){};

// Clone virtual function
__host__ virtual CKernel* clone() const = 0;

// Virtual functions to initialize GCF
__host__ virtual void initializeGCF(){this->gcf = this->clone();};
__host__ virtual void initializeGCF(int m, int n, float dx, float dy){
        this->gcf = this->clone();
        this->gcf->setmn(m, n);
        this->gcf->setSigmas(dx, dy);
        this->gcf->setW(m);
        this->gcf->buildGCF();
};
__host__ virtual void initializeGCF(int m, int n, float dx, float dy, float w){
        this->gcf = this->clone();
        this->gcf->setmn(m, n);
        this->gcf->setSigmas(dx, dy);
        this->gcf->setW(w);
        this->gcf->buildGCF();
};


__host__ CKernel()
{
        this->amp = 1.0f;
        this->m = 7;
        this->n = 7;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, CKernel *gcf)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, Io *imageHandler)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, Io *imageHandler, CKernel *gcf)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, CKernel *gcf)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, Io *imageHandler)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, Io *imageHandler, CKernel *gcf)
{
        this->amp = 1.0f;
        this->m = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, float w)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float w, CKernel *gcf)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = gcf;
        this->name = "";
};


__host__ CKernel(int m, int n, float w, Io *imageHandler)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float w, Io *imageHandler, CKernel *gcf)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = 1.0f;
        this->sigma_y = 1.0f;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, float w)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, float w, CKernel *gcf)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
        this->gcf = gcf;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, float w, Io *imageHandler)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = NULL;
        this->name = "";
};

__host__ CKernel(int m, int n, float dx, float dy, float w, Io *imageHandler, CKernel *gcf)
{
        this->amp = 1.0f;
        this->n = m;
        this->n = n;
        this->sigma_x = dx;
        this->sigma_y = dy;
        this->x0 = 0.0f;
        this->y0 = 0.0f;
        this->w = w;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
        this->gcf = gcf;
        this->name = "";
};

__host__ ~CKernel() {
        this->kernel.clear();
        this->freeGPUKernel();
        if(this->gcf != NULL){
          this->gcf->kernel.clear();
          this->gcf->freeGPUKernel();
        }
};

__host__ float getAmp(){
        return this->amp;
};
__host__ int getm(){
        return this->m;
};
__host__ int getn(){
        return this->n;
};

__host__ int2 getMN(){
        int2 ret;
        ret.x = this->m;
        ret.y = this->n;
};

__host__ int getSigmaX(){
        return this->sigma_x;
};
__host__ int getSigmaY(){
        return this->sigma_y;
};
__host__ int getSupportX(){
        return this->support_x;
};
__host__ int getSupportY(){
        return this->support_y;
};

__host__ int getGPUID(){
        return this->gpu_id;
};

__host__ float getW(){
        return this->w;
};

__host__ float getX0(){
        return this->x0;
};

__host__ float getY0(){
        return this->y0;
};

__host__ float2 getCenter(){
        float2 ret;
        ret.x = this->x0;
        ret.y = this->y0;
        return ret;
};

__host__ float getKernelValue(int i, int j)
{
        return this->kernel[this->n * i + j];
};
__host__ std::vector<float> getKernel()
{
        return this->kernel;
};
__host__ float* getKernelPointer()
{
        return this->kernel.data();
};
__host__ float* getGPUKernel()
{
        return this->gpu_kernel;
};
__host__ std::string getName(){
        return this->name;
};
__host__ Io* getImageHandler(){
        return this->ioImageHandler;
};

__host__ void setName(std::string name){
        this->name = name;
};

__host__ void setAmp(float amp){
        this->amp = amp;
};

__host__ void setCenter(float x0, float y0){
        this->x0 = x0; this->y0 = y0;
};

__host__  void setmn(int m, int n){
        this->m = m; this->n = n;
        this->setm_times_n();
        this->setSupports();
};

__host__  void setSigmas(float dx, float dy){
        this->sigma_x = dx; this->sigma_y = dy;
};

__host__  void setW(float w){
        this->w = w;
};

__host__ void setIoImageHandler(Io *imageHandler){
        this->ioImageHandler = imageHandler;
};

__host__ void setGPUID(int gpu_id){
        this->gpu_id = gpu_id;
};

__host__ void printGCF(){
        if(this->gcf->getImageHandler() != NULL && this->gcf->getImageHandler()->getPrintImages())
            this->gcf->getImageHandler()->printImage(this->gcf->getKernelPointer(), "GCF.fits", "", 0, 0, 1.0f, this->gcf->getm(), this->gcf->getn(), false);
        else
            std::cout << "No IO Image object to print the image or print images has been set on false" << std::endl;
};

__host__ void printCKernel(){
        if(this->ioImageHandler != NULL && this->ioImageHandler->getPrintImages())
            this->ioImageHandler->printImage(this->getKernelPointer(), "ckernel.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), false);
        else
            std::cout << "No IO Image object to print the image or print images has been set on false" << std::endl;
};

__host__ void printGPUCKernel(){
        if(this->ioImageHandler != NULL && this->ioImageHandler->getPrintImages())
            this->ioImageHandler->printImage(this->getGPUKernel(), "ckernel_gpu.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), true);
        else
            std::cout << "No IO Image object to print the image or print images has been set on false" << std::endl;
};

__host__ void printGPUGCF(){
        if(this->gcf->getImageHandler() != NULL && this->gcf->getImageHandler()->getPrintImages())
            this->gcf->getImageHandler()->printImage(this->gcf->getGPUKernel(), "ckernel_gpu.fits", "", 0, 0, 1.0f, this->gcf->getm(), this->gcf->getn(), true);
        else
            std::cout << "No IO Image object to print the image or print images has been set on false" << std::endl;
};

private:
int m_times_n;

__host__  void setm_times_n(){
        this->m_times_n = this->m * this->n;
};

__host__  void setSupports(){
        this->support_x = floor(this->m/2.0f);
        this->support_y = floor(this->m/2.0f);
};

__host__ void freeGPUKernel()
{
        cudaFree(this->gpu_kernel);
};

protected:
int m;     //size of the kernel
int n;     //size of the kernel
int support_x;
int support_y;
int gpu_id = 0;
float amp;
float x0;
float y0;
float sigma_x;
float sigma_y;
float w;
std::vector<float> kernel;
float *gpu_kernel;
// Image IO handler to print images if necessary
Io *ioImageHandler = NULL;
// Gridding correction function (GCF)
CKernel *gcf = NULL;
// Name of the Convolution Kernel
std::string name;


__host__ void setKernelMemory()
{
        this->kernel.resize(this->m_times_n);
        cudaSetDevice(this->gpu_id);
        checkCudaErrors(cudaMalloc(&this->gpu_kernel, sizeof(float) * this->m_times_n));
        checkCudaErrors(cudaMemset(this->gpu_kernel, 0, sizeof(float) * this->m_times_n));
};

__host__ void copyKerneltoGPU()
{
        cudaSetDevice(this->gpu_id);
        checkCudaErrors(cudaMemcpy(this->gpu_kernel, this->kernel.data(), sizeof(float) * this->m_times_n, cudaMemcpyHostToDevice));
};

};
#endif //CKERNEL_CUH
