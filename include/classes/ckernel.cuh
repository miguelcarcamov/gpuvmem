#ifndef CKERNEL_CUH
#define CKERNEL_CUH

class CKernel
{
public:
__host__ virtual void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y) = 0;
__host__ __device__ CKernel::CKernel()
{
        this->m = 7;
        this->n = 7;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ __device__ CKernel::CKernel(int m, int n)
{
        this->m = m;
        this->n = n;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ CKernel::CKernel(int m, int n, Io *imageHandler)
{
        this->m = m;
        this->n = n;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
};

__host__ __device__ CKernel::CKernel(int m, int n, float w1)
{
        this->n = m;
        this->n = n;
        this->w1 = w1;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ CKernel::CKernel(int m, int n, float w1, Io *imageHandler)
{
        this->n = m;
        this->n = n;
        this->w1 = w1;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
};

__host__ __device__ CKernel::CKernel(int m, int n, float w1, float w2)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ CKernel::CKernel(int m, int n, float w1, float w2, Io *imageHandler)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
};



__host__ __device__ CKernel::CKernel(float w1, float w2, float angle, int m, int n)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = 2.0;
        this->angle = angle;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ CKernel::CKernel(float w1, float w2, float angle, int m, int n, Io *imageHandler)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = 2.0;
        this->angle = angle;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
};

__host__ __device__ CKernel::CKernel(float w1, float w2, float alpha, float angle, int m, int n)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = alpha;
        this->angle = angle;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = NULL;
};

__host__ CKernel::CKernel(float w1, float w2, float alpha, float angle, int m, int n, Io *imageHandler)
{
        this->m = m;
        this->n = n;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = alpha;
        this->angle = angle;
        this->setm_times_n();
        this->setSupports();
        this->ioImageHandler = imageHandler;
};

__host__ ~CKernel() {
        this->kernel.clear();
        this->freeGPUKernel();
};

__host__ __device__ int getm(){
        return this->m;
};
__host__ __device__ int getn(){
        return this->n;
};
__host__ __device__ int getSupportX(){
        return this->support_x;
};
__host__ __device__ int getSupportY(){
        return this->support_y;
};
__host__ __device__ float getW1(){
        return this->w1;
};
__host__ __device__ float getW2(){
        return this->w2;
};
__host__ __device__ float getAlpha(){
        return this->alpha;
};
__host__ __device__ float getAngle(){
        return this->angle;
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
__host__ __device__ void setmn(int m, int n){
        this->m = m; this->n = n;
        this->setm_times_n();
        this->setSupports();
};

__host__ __device__ void setW1(float w1){
        this->w1 = w1;
};
__host__ __device__ void setW2(float w2){
        this->w2 = w2;
};
__host__ __device__ void setAlpha(float alpha){
        this->alpha = alpha;
};
__host__ __device__ void setAngle(float angle){
        this->angle = angle;
};
__host__ void setIoImageHandler(Io *imageHandler){
        this->ioImageHandler = imageHandler;
};
__host__ void printCKernel(){
        this->ioImageHandler->printImage(this->getKernelPointer(), "ckernel.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), false);
};

__host__ void printGPUCKernel(){
        this->ioImageHandler->printImage(this->getGPUKernel(), "ckernel_gpu.fits", "", 0, 0, 1.0f, this->getm(), this->getn(), true);
};

private:
int m_times_n;

__host__ __device__ void setm_times_n(){
        this->m_times_n = this->m * this->n;
};

__host__ __device__ void setSupports(){
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
float w1;
float w2;
float alpha;
float angle;
std::vector<float> kernel;
float *gpu_kernel;
Io *ioImageHandler = NULL;

__host__ void setKernelMemory(){
        this->kernel.resize(this->m_times_n);
        checkCudaErrors(cudaMalloc(&this->gpu_kernel, sizeof(float) * this->m_times_n));
        checkCudaErrors(cudaMemset(this->gpu_kernel, 0, sizeof(float) * this->m_times_n));
};

__host__ void copyKerneltoGPU(){
        checkCudaErrors(cudaMemcpy(this->gpu_kernel, this->kernel.data(), sizeof(float) * this->m_times_n, cudaMemcpyHostToDevice));
};

__host__ __device__ float gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha)
{
        float radius_x = distance(x, 0.0f, x0, 0.0f);
        float val = radius_x/(w*sigma);
        float val_alpha = powf(val, alpha);
        float G = amp*expf(-val_alpha);

        return G;
};

__host__ __device__ float gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float radius_x = distance(x, 0.0f, x0, 0.0f);
        float radius_y = distance(0.0f, y, 0.0f, y0);

        if(radius_x < w*sigma_x && radius_y < w*sigma_y) {
                float fx = radius_x/(w*sigma_x);
                float fy = radius_y/(w*sigma_y);

                float val_x = powf(fx, alpha);
                float val_y = powf(fy, alpha);
                float G = amp*expf(-1.0f*(val_x+val_y));

                return G;
        }else
                return 0.0f;
};

__host__ __device__ float sincf(float x)
{
        float value;
        if(x==0.0f)
                value = 1.0f;
        else
                value = sinf(PI*x)/(PI*x);

        return value;
};

__host__ __device__ float sinc1D(float amp, float x, float x0, float sigma, float w)
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

__host__ __device__ float gaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha)
{
        return amp*gaussian1D(1.0f, x, x0, sigma, w1, alpha)*sinc1D(1.0f, x, x0, sigma, w2);
};

__host__ __device__ float sinc2D(float amp, float x, float x0, float y, float y0, float sigma_x, float sigma_y, float w)
{
        float s_x = sinc1D(1.0f, x, x0, sigma_x, w);
        float s_y = sinc1D(1.0f, y, y0, sigma_y, w);
        return amp*s_x*s_y;
};

__host__ __device__ float gaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha)
{
        float G = gaussian2D(1.0f, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        float S = sinc2D(1.0f, x, x0, y, y0, sigma_x, sigma_y, w2);
        //printf("Gaussian :%f - Sinc: %f\n", G, S);
        return amp*G*S;
};

__host__ __device__ float pillBox1D(float amp, float x, float limit)
{
        if(fabs(x) < limit)
                return amp;
        else
                return 0.0f;
};

__host__ __device__ float pillBox2D(float amp, float x, float y, float limit_x, float limit_y)
{
        return pillBox1D(amp, x, limit_x)*pillBox1D(amp, y, limit_y);
};

__host__ __device__ float pswf_11D_func(float nu)
{
        float nu_end;
        float dnusq, top, bottom;
        int idx;

        const float mat_p[2][5] = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};

        const float mat_q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                {1.0000000e0, 9.599102e-1, 2.918724e-1}};

        float n_nu = fabsf(nu);
        float res = 0.0f;
        if(n_nu > 1.0f)
                res = 0.0f;
        else{
                nu_end = 0.0f;
                idx = 0;
                if(n_nu >= 0.0f && n_nu < 0.75) {
                        idx = 0;
                        nu_end = 0.75f;
                }else{
                        idx = 1;
                        nu_end = 1.0f;
                }

                dnusq = n_nu*n_nu - nu_end*nu_end;
                top = mat_p[idx][0];
                bottom = mat_q[idx][0];

                for(int i=1; i<5; i++) {
                        top += mat_p[idx][i] * powf(dnusq, i);
                }

                for(int i=1; i<3; i++) {
                        bottom += mat_q[idx][i] * powf(dnusq, i);
                }

                if(bottom > 0.0f) {
                        res = top/bottom;
                }

        }
        return res;
};

__host__ __device__ float pswf_11D(float amp, float x, float x0, float sigma, float w)
{
        float nu, pswf, nu_sq, val;
        float radius = distance(x, 0.0f, x0, 0.0f);

        nu =  radius/(w*sigma);
        if(nu==0.0f) {
                val = 1.0f;
        }else{
                pswf = pswf_11D_func(nu);
                nu_sq = nu * nu;
                val = amp*(1.0f-nu_sq)*pswf;
        }
        return val;
};

__host__ __device__ float pswf_12D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w)
{
        float xval = pswf_11D(1.0f, x, x0, sigma_x, w);
        float yval = pswf_11D(1.0f, y, y0, sigma_y, w);
        float val = amp*xval*yval;
        return val;
};

};
#endif //CKERNEL_CUH
