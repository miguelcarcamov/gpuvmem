#ifndef CKERNEL_CUH
#define CKERNEL_CUH

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
__host__ virtual float * getGCFGPUValues(){return this->gcf->getGPUKernel();};
__host__ virtual std::vector<float> getGCFCPUValues(){return this->gcf->getKernel();};
__host__ virtual float* getGCFCPUPointer(){return this->gcf->getKernelPointer();};
__host__ virtual void printGCFCPU(){return this->gcf->printCKernel();};
__host__ virtual void printGCFGPU(){return this->gcf->printGPUCKernel();};

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
__host__ void printCKernel(){
        if(this->ioImageHandler != NULL && this->ioImageHandler->getPrintImages())
            this->ioImageHandler->printImage(this->getKernelPointer(), "ckernel.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), false);
        else
            std::cout << "No IO Image object to print the image" << std::endl;
};

__host__ void printGPUCKernel(){
        if(this->ioImageHandler != NULL && this->ioImageHandler->getPrintImages())
            this->ioImageHandler->printImage(this->getGPUKernel(), "ckernel_gpu.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), true);
        else
            std::cout << "No IO Image object to print the image" << std::endl;
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


__host__ void setKernelMemory(){
        this->kernel.resize(this->m_times_n);
        checkCudaErrors(cudaMalloc(&this->gpu_kernel, sizeof(float) * this->m_times_n));
        checkCudaErrors(cudaMemset(this->gpu_kernel, 0, sizeof(float) * this->m_times_n));
};

__host__ void copyKerneltoGPU(){
        checkCudaErrors(cudaMemcpy(this->gpu_kernel, this->kernel.data(), sizeof(float) * this->m_times_n, cudaMemcpyHostToDevice));
};

};
#endif //CKERNEL_CUH
