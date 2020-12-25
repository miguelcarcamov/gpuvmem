#include "gl1norm.cuh"

extern long M, N;
extern int image_count;
extern float * penalizators;
extern int nPenalizators;

GL1Norm::GL1Norm(){
        this->name = "G L1-Norm";
        this->prior = NULL;
        this->normalization_factor = 1.0f;
};

GL1Norm::GL1Norm(std::vector<float> prior){
        this->name = "G L1-Norm";
        this->normalization_factor = 1.0f;
        checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float)*M*N));
        checkCudaErrors(cudaMemcpy(this->prior, prior.data(), M*N, cudaMemcpyHostToDevice));
};

GL1Norm::GL1Norm(float *prior, float normalization_factor){
        this->name = "G L1-Norm";
        this->prior = prior;
        this->normalization_factor = normalization_factor;
        this->normalizePrior();
};

GL1Norm::GL1Norm(std::vector<float> prior, float normalization_factor){
        this->name = "G L1-Norm";
        this->normalization_factor = normalization_factor;
        checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float)*M*N));
        checkCudaErrors(cudaMemcpy(this->prior, prior.data(), M*N, cudaMemcpyHostToDevice));
        this->normalizePrior();
};

GL1Norm::GL1Norm(float *prior){
        this->name = "G L1-Norm";
        this->prior = prior;
        this->normalization_factor = 1.0f;
};

GL1Norm::~GL1Norm() {
        cudaFree(this->prior);
};

float GL1Norm::getNormalizationFactor(){
        return this->normalization_factor;
};

void GL1Norm::setNormalizationFactor(float normalization_factor){
        this->normalization_factor = normalization_factor;
};

void GL1Norm::setPrior(float *prior){
        cudaFree(this->prior);
        this->prior = prior;
};

float GL1Norm::calcFi(float *p)
{
        float result = 0.0;
        this->set_fivalue(SGEntropy(p, device_S, this->prior, penalization_factor, mod, order, imageIndex));
        result = (penalization_factor)*( this->get_fivalue() );
        return result;
};

void GL1Norm::calcGi(float *p, float *xi)
{
        DGEntropy(p, device_DS, this->prior, penalization_factor, mod, order, imageIndex);
};


void GL1Norm::restartDGi()
{
        checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float)*M*N));
};

void GL1Norm::addToDphi(float *device_dphi)
{
        linkAddToDPhi(device_dphi, device_DS, imageToAdd);
};

void GL1Norm::setSandDs(float *S, float *Ds)
{
        cudaFree(this->device_S);
        cudaFree(this->device_DS);
        this->device_S = S;
        this->device_DS = Ds;
};

void GL1Norm::normalizePrior()
{
        normalizeImage(this->prior, this->normalization_factor);
};

namespace {
Fi* CreateGL1Norm()
{
        return new GL1Norm;
}
const std::string name = "GL1Norm";
const bool RegisteredGL1Norm = registerCreationFunction<Fi, std::string>(name, CreateGL1Norm);
const bool RegisteredGL1NormInt = registerCreationFunction<Fi, int>(0, CreateGL1Norm);
};
