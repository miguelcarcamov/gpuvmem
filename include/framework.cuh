#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <ctgmath>
#include <string>
#include <functional>
#include <numeric>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math_constants.h>
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include "MSFITSIO.cuh"
#include "copyrightwarranty.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

typedef struct varsPerGPU {
        float *device_chi2;
        float *device_dchi2;
        cufftHandle plan;
        cufftComplex *device_I_nu;
        cufftComplex *device_V;
}varsPerGPU;

typedef struct variables {
        char *input;
        char *output;
        char *inputdat;
        char *modin;
        char *ofile;
        char *path;
        char *output_image;
        char *gpus;
        char *initial_values;
        char *penalization_factors;
        int blockSizeX;
        int blockSizeY;
        int blockSizeV;
        int it_max;
        int gridding;
        float noise;
        float noise_cut;
        float randoms;
        float eta;
        float nu_0;
        float robust_param;
        float threshold;
} Vars;

typedef struct functionMap
{
        void (*newP)(float*, float*, float, int);
        void (*evaluateXt)(float*, float*, float*, float, int);
} imageMap;

class VirtualImageProcessor
{
public:
virtual void clip(float *I) = 0;
virtual void clipWNoise(float *I) = 0;
virtual void apply_beam(cufftComplex *image, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float freq, int primary_beam) = 0;
virtual void calculateInu(cufftComplex *image, float *I, float freq) = 0;
virtual void chainRule(float *I, float freq) = 0;
virtual void configure(int I) = 0;
protected:
float *chain;
cufftComplex *fg_image;
int image_count;
};


template<class T>
class Singleton
{
public:
static T& Instance()
{
        static T instance;
        return instance;
}
private:
Singleton(){
};
Singleton(T const&)         = delete;
void operator=(T const&) = delete;
};

class Fi
{
public:
virtual float calcFi(float *p) = 0;
virtual void calcGi(float *p, float *xi) = 0;
virtual void restartDGi() = 0;
virtual void addToDphi(float *device_dphi) = 0;
virtual void configure(int penalizatorIndex, int imageIndex, int imageToAdd) = 0;

float get_fivalue(){
        return this->fi_value;
};
void set_fivalue(float fi){
        this->fi_value = fi;
};
void setPenalizationFactor(float p){
        this->penalization_factor = p;
};
void setInu(cufftComplex *Inu){
        this->Inu = Inu;
}
cufftComplex *getInu(){
        return this->Inu;
}
void setS(float *S){
        cudaFree(device_S); this->device_S = S;
};
void setDS(float *DS){
        cudaFree(device_DS); this->device_DS = DS;
};
virtual float calculateSecondDerivate() = 0;
float getPenalizationFactor(){
        return this->penalization_factor;
}
protected:
float fi_value;
float *device_S;
float *device_DS;
float penalization_factor = 1;
int imageIndex;
int mod;
int order;
cufftComplex * Inu = NULL;
int imageToAdd;
};


class Image
{
public:
Image(float *image, int image_count){
        this->image = image; this->image_count = image_count;
};
int getImageCount(){
        return image_count;
};
float *getImage(){
        return image;
};
float *getErrorImage(){
        return error_image;
};
imageMap *getFunctionMapping(){
        return functionMapping;
};
void setImageCount(int i){
        this->image_count = i;
};
void setErrorImage(float *f){
        this->error_image = f;
};
void setImage(float *i){
        this->image = i;
};
void setFunctionMapping(imageMap *f){
        this->functionMapping = f;
};
private:
int image_count;
float *image;
float *error_image;
imageMap *functionMapping;
};

class Visibilities
{
public:

void setMSDataset(std::vector<MSDataset> d){
        this->datasets = d;
};
void setTotalVisibilities(int t){
        this->total_visibilities = t;
};

void setNDatasets(int t){
        this->ndatasets = t;
};

void setMaxNumberVis(int t){
        this->max_number_vis = t;
};

std::vector<MSDataset> getMSDataset(){
        return datasets;
};
int getTotalVisibilities(){
        return total_visibilities;
};

int getMaxNumberVis(){
        return max_number_vis;
};

int getNDatasets(){
        return ndatasets;
};

private:
std::vector<MSDataset> datasets;
int ndatasets;
int total_visibilities;
int max_number_vis;
};

class Error
{
public:
virtual void calculateErrorImage(Image *I, Visibilities *v) = 0;
};

class Io
{
public:
virtual canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag) = 0;
virtual void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding) = 0;
virtual void IocopyMS(char const *infile, char const *outfile) = 0;
virtual void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag) = 0;
virtual void IocloseCanvas(fitsfile *canvas) = 0;
virtual void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, bool isInGPU) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) = 0;
void setPrintImagesPath(char * pip){
        this->printImagesPath = pip;
};
protected:
int *iteration;
char *printImagesPath;
};

class ObjectiveFunction
{
public:
ObjectiveFunction(){
};
void addFi(Fi *fi){
        if(fi->getPenalizationFactor()) {
                fis.push_back(fi);
                fi_values.push_back(0.0f);
        }
};
//virtual void print() = 0;
float calcFunction(float *p)
{
        float value = 0.0;
        int fi_value_count = 0;
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                float iterationValue = (*it)->calcFi(p);
                fi_values[fi_value_count] = (*it)->get_fivalue();
                value += iterationValue;
                fi_value_count++;
        }

        return value;
};

void calcGradient(float *p, float *xi)
{
        if(print_images) {
                if(IoOrderIterations == NULL) {
                        io->IoPrintOptImageIteration(p,"I_nu_0","JY/PIXEL",0, true);
                        io->IoPrintOptImageIteration(p,"alpha","",1, true);
                }else{
                        (IoOrderIterations)(p, io);
                }
        }
        restartDPhi();
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->calcGi(p, xi);
                (*it)->addToDphi(dphi);
        }
        phiStatus = 1;
        copyDphiToXi(xi);
};

void restartDPhi()
{
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->restartDGi();
        }
        checkCudaErrors(cudaMemset(dphi, 0, sizeof(float)*M*N*image_count));
}

void copyDphiToXi(float *xi)
{
        checkCudaErrors(cudaMemcpy(xi, dphi, sizeof(float)*M*N*image_count, cudaMemcpyDeviceToDevice));
}

std::vector<Fi*> getFi(){
        return this->fis;
};
void setN(long N){
        this->N = N;
}
void setM(long M){
        this->M = M;
}
void setImageCount(int I){
        this->image_count = I;
}
void setIo(Io *i){
        this->io = i;
};
void setPrintImages(int i){
        this->print_images = i;
};
void setIoOrderIterations(void (*func)(float *I, Io *io)){
        this->IoOrderIterations = func;
};
void configure(long N, long M, int I)
{
        setN(N);
        setM(M);
        setImageCount(I);
        checkCudaErrors(cudaMalloc((void**)&dphi, sizeof(float)*M*N*I));
        checkCudaErrors(cudaMemset(dphi, 0, sizeof(float)*M*N*I));
}
std::vector<float> get_fi_values(){
        return this->fi_values;
}
private:
std::vector<Fi*> fis;
std::vector<float> fi_values;
Io *io = NULL;
float *dphi;
int phiStatus = 1;
int flag = 0;
long N = 0;
long M = 0;
int print_images = 0;
void (*IoOrderIterations)(float *I, Io *io) = NULL;
int image_count = 1;
};

class Optimizator
{
public:
__host__ virtual void allocateMemoryGpu() = 0;
__host__ virtual void deallocateMemoryGpu() = 0;
__host__ virtual void optimize() = 0;
//__host__ virtual void configure() = 0;
__host__ void setImage(Image *image){
        this->image = image;
};
__host__ void setObjectiveFunction(ObjectiveFunction *of){
        this->of = of;
};
void setFlag(int flag){
        this->flag = flag;
};
ObjectiveFunction* getObjectiveFuntion(){
        return this->of;
};
protected:
ObjectiveFunction *of;
Image *image;
int flag;
};



class Filter
{
public:
virtual void applyCriteria(Visibilities *v) = 0;
virtual void configure(void *params) = 0;
};

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
};

__host__ __device__ CKernel::CKernel(int m, int n)
{
        this->m = m;
        this->n = n;
        this->alpha = 2;
        this->angle = 0.0;
        this->setm_times_n();
        this->setSupports();
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

private:
int m_times_n;

__host__ __device__ void setm_times_n(){
        this->m_times_n = this->m * this->n;
};

__host__ __device__ void setSupports(){
        this->support_x = floor(this->m/2.0f);
        this->support_y = floor(this->m/2.0f);
};


protected:
int m; //size of the kernel
int n; //size of the kernel
int support_x;
int support_y;
float w1;
float w2;
float alpha;
float angle;
std::vector<float> kernel;

__host__ void setKernelMemory(){
        this->kernel.resize(this->m_times_n);
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
        pswf = pswf_11D_func(nu);
        nu_sq = nu * nu;
        val = amp*(1.0f-nu_sq)*pswf;
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
//Implementation of Factories
class Synthesizer
{
public:
__host__ virtual void run() = 0;
__host__ virtual void setOutPut(char * FileName) = 0;
__host__ virtual void setDevice() = 0;
__host__ virtual void unSetDevice() = 0;
__host__ virtual std::vector<std::string> countAndSeparateStrings(char *input) = 0;
__host__ virtual void configure(int argc, char **argv) = 0;
__host__ virtual void applyFilter(Filter *filter) = 0;
__host__ virtual void writeImages() = 0;
__host__ virtual void clearRun() = 0;
__host__ virtual void writeResiduals() = 0;
__host__ void setOptimizator(Optimizator *min){
        this->optimizator = min;
};
__host__ void setVisibilities(Visibilities * v){
        this->visibilities = v;
};
__host__ void setIoHandler(Io *handler){
        this->iohandler = handler;
};
__host__ void setError(Error *e){
        this->error = e;
};
__host__ void setOrder(void (*func)(Optimizator *o,Image *I)){
        this->Order = func;
};
Image *getImage(){
        return image;
};
void setImage(Image *i){
        this->image = i;
};
void setIoOrderEnd(void (*func)(float *I, Io *io)){
        this->IoOrderEnd = func;
};
void setIoOrderError(void (*func)(float *I, Io *io)){
        this->IoOrderError = func;
};
void setIoOrderIterations(void (*func)(float *I, Io *io)){
        this->IoOrderIterations = func;
};

Optimizator* getOptimizator(){
        return this->optimizator;
};

__host__ void setGriddingKernel(CKernel *ckernel){
        this->ckernel = ckernel;
}
protected:
cufftComplex *device_I;
Image *image;
Optimizator *optimizator;
CKernel *ckernel;
Io *iohandler = NULL;
Visibilities *visibilities;
Error *error = NULL;
int griddingMode = 0;
void (*Order)(Optimizator *o, Image *I) = NULL;
int imagesChanged = 0;
void (*IoOrderIterations)(float *I, Io *io) = NULL;
void (*IoOrderEnd)(float *I, Io *io) = NULL;
void (*IoOrderError)(float *I, Io *io) = NULL;
};



class SynthesizerFactory
{
public:
typedef Synthesizer* (*CreateSynthesizerCallback)();
private:
typedef std::map<int, CreateSynthesizerCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterSynthesizer(int SynthesizerId, CreateSynthesizerCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(SynthesizerId, CreateFn)).second;
};

bool UnregisterSynthesizer(int SynthesizerId)
{
        return callbacks_.erase(SynthesizerId) == 1;
};

Synthesizer* CreateSynthesizer(int SynthesizerId)
{
        CallbackMap::const_iterator i = callbacks_.find(SynthesizerId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Synthesizer ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class FiFactory
{
public:
typedef Fi* (*CreateFiCallback)();
private:
typedef std::map<int, CreateFiCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterFi(int FiId, CreateFiCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(FiId, CreateFn)).second;
};

bool UnregisterFi(int FiId)
{
        return callbacks_.erase(FiId) == 1;
};

Fi* CreateFi(int FiId)
{
        CallbackMap::const_iterator i = callbacks_.find(FiId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Fi ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class OptimizatorFactory
{
public:
typedef Optimizator* (*CreateOptimizatorCallback)();
private:
typedef std::map<int, CreateOptimizatorCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterOptimizator(int OptimizatorId, CreateOptimizatorCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(OptimizatorId, CreateFn)).second;
};

bool UnregisterOptimizator(int OptimizatorId)
{
        return callbacks_.erase(OptimizatorId) == 1;
};

Optimizator* CreateOptimizator(int OptimizatorId)
{
        CallbackMap::const_iterator i = callbacks_.find(OptimizatorId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Optimizator ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class CKernelFactory
{
public:
typedef CKernel* (*CreateCKernelCallback)();
private:
typedef std::map<int, CreateCKernelCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterCKernel(int CKernelId, CreateCKernelCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(CKernelId, CreateFn)).second;
};

bool UnregisterCKernel(int CKernelId)
{
        return callbacks_.erase(CKernelId) == 1;
};

CKernel* CreateCKernel(int CKernelId)
{
        CallbackMap::const_iterator i = callbacks_.find(CKernelId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown CKernel ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class FilterFactory
{
public:
typedef Filter* (*CreateFilterCallback)();
private:
typedef std::map<int, CreateFilterCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterFilter(int FilterId, CreateFilterCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(FilterId, CreateFn)).second;
};

bool UnregisterFilter(int FilterId)
{
        return callbacks_.erase(FilterId) == 1;
};

Filter* CreateFilter(int FilterId)
{
        CallbackMap::const_iterator i = callbacks_.find(FilterId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Filter ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class ObjectiveFunctionFactory
{
public:
typedef ObjectiveFunction* (*CreateObjectiveFunctionCallback)();
private:
typedef std::map<int, CreateObjectiveFunctionCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterObjectiveFunction(int ObjectiveFunctionId, CreateObjectiveFunctionCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(ObjectiveFunctionId, CreateFn)).second;
};

bool UnregisterObjectiveFunction(int ObjectiveFunctionId)
{
        return callbacks_.erase(ObjectiveFunctionId) == 1;
};

ObjectiveFunction* CreateObjectiveFunction(int ObjectiveFunctionId)
{
        CallbackMap::const_iterator i = callbacks_.find(ObjectiveFunctionId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown ObjectiveFunction ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class IoFactory
{
public:
typedef Io* (*CreateIoCallback)();
private:
typedef std::map<int, CreateIoCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterIo(int IoId, CreateIoCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(IoId, CreateFn)).second;
};

bool UnregisterIo(int IoId)
{
        return callbacks_.erase(IoId) == 1;
};

Io* CreateIo(int IoId)
{
        CallbackMap::const_iterator i = callbacks_.find(IoId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Io ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class ErrorFactory
{
public:
typedef Error* (*CreateErrorCallback)();
private:
typedef std::map<int, CreateErrorCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterError(int ErrorId, CreateErrorCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(ErrorId, CreateFn)).second;
};

bool UnregisterError(int ErrorId)
{
        return callbacks_.erase(ErrorId) == 1;
};

Error* CreateError(int ErrorId)
{
        CallbackMap::const_iterator i = callbacks_.find(ErrorId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Error ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

namespace {
ObjectiveFunction* CreateObjectiveFunction()
{
        return new ObjectiveFunction;
}
const int ObjectiveFunctionId = 0;
const bool Registered = Singleton<ObjectiveFunctionFactory>::Instance().RegisterObjectiveFunction(ObjectiveFunctionId, CreateObjectiveFunction);
};

#endif
