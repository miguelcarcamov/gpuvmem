#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <ctgmath>
#include <string>
#include <functional>
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
virtual void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)= 0;
virtual void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option)=0;
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
        if(fi->getPenalizationFactor()){
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
                        io->IoPrintOptImageIteration(p,"I_nu_0","JY/PIXEL",0);
                        io->IoPrintOptImageIteration(p,"alpha","",1);
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
std::vector<float> get_fi_values(){ return this->fi_values; }
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
CKernel()
{
        this->M = 6;
        this->N = 6;
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2;

};


CKernel(float dx, float dy, int M, int N)
{
        this->M = M;
        this->N = N;
        this->dx = dx;
        this->dy = dy;
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2;
        this->setM_times_N();
};


CKernel::CKernel(float dx, float dy, float w1, float w2, float alpha, int M, int N)
{
        this->M = M;
        this->N = N;
        this->dx = dx;
        this->dy = dy;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = alpha;
        this->setM_times_N();
};
float getdx(){return this->dx;};
float getdy(){return this->dy;};
int2 getMN(){int2 val; val.x = this->M; val.y = this->N; return val;};
float getW1(){return this->w1;};
float getW2(){return this->w2;};
float getAlpha(){return this->alpha;};
void setdxdy(float dx, float dy){this->dx = dx; this->dx = dx;};
void setMN(int M, int N){this->M = M; this->N = N;};
void setW1(float w1){this->w1 = w1;};
void setW2(float w2){this->w2 = w2;};
void setAlpha(float alpha){this->alpha = alpha;};
float run(float deltau, float deltav){return 1.0f;};
private:
void setM_times_N(){this->M_times_N = this->M * this->N;};
__host__ __device__ float ellipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle)
{
        float x_i = x-x0;
        float y_i = y-y0;
        float cos_angle, sin_angle;
        sincos(angle, &sin_angle, &cos_angle);
        float sin_angle_2 = sin(2.0*angle);
        float a = (cos_angle*cos_angle)/(2.0*sigma_x*sigma_x) + (sin_angle*sin_angle)/(2.0*sigma_y*sigma_y);
        float b = sin_angle_2/(2.0*sigma_x*sigma_x) - sin_angle_2/(2.0*sigma_y*sigma_y);
        float c = (sin_angle*sin_angle)/(2.0*sigma_x*sigma_x) + (cos_angle*cos_angle)/(2.0*sigma_y*sigma_y);
        float G = amp*exp(-a*x_i*x_i - b*x_i*y_i - c*y_i*y_i);

        return G;
};
__host__ __device__ float gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float x_i = x-x0;
        float y_i = y-y0;

        float num_x = pow(x_i, alpha);
        float num_y = pow(y_i, alpha);

        float den_x = 2.0*pow(w*sigma_x,alpha);
        float den_y = 2.0*pow(w*sigma_y,alpha);

        float val_x = num_x/den_x;
        float val_y = num_y/den_y;
        float G = amp*exp(-val_x-val_y);

        return G;
};
__host__ __device__ float gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha)
{
        float x_i = x-x0;
        float val = abs(x_i)/(w*sigma);
        float val_alpha = pow(val, alpha);
        float G = amp*exp(-val_alpha);

        return G;
};
__host__ __device__ float sinc1D(float amp, float x, float x0, float sigma, float w)
{
        float s = 1.0f/*amp*sinc((x-x0)/(w*sigma))*/;
        return s;
};
__host__ __device__ float gaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha)
{
        return amp*gaussian1D(1.0, x, x0, sigma, w1, alpha)*sinc1D(1.0, x, x0, sigma, w2);
};
__host__ __device__ float sinc2D(float amp, float x, float x0, float y, float y0, float sigma_x, float sigma_y, float w)
{
        float s_x = sinc1D(1.0, x, x0, sigma_x, w);
        float s_y = sinc1D(1.0, y, y0, sigma_y, w);
        return amp*s_x*s_y;
};
__host__ __device__ float gaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha)
{
        float G = gaussian2D(1.0, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        float S = sinc2D(1.0, x, x0, y, y0, sigma_x, sigma_y, w2);
        return amp*G*S;
};
int M;
int N;
float w1;
float w2;
float alpha;
float dx;
float dy;
float M_times_N;
};

//Implementation of Factory
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
