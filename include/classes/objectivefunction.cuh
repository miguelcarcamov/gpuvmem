#ifndef OBJECTIVEFUNCTION_CUH
#define OBJECTIVEFUNCTION_CUH

#include <factory.cuh>

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

void calcGradient(float *p, float *xi, int iter)
{
        if(io->getPrintImages()) {
                if(IoOrderIterations == NULL) {
                        io->printImageIteration(p, "I_nu_0", "JY/PIXEL", iter, 0, true);
                        io->printImageIteration(p, "alpha", "JY/PIXEL", iter, 0, true);
                }else{
                        (IoOrderIterations)(p, io);
                }
        }
        restartDPhi();
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->setIteration(iter);
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
void (*IoOrderIterations)(float *I, Io *io) = NULL;
int image_count = 1;
};

namespace {
ObjectiveFunction* CreateObjectiveFunction()
{
        return new ObjectiveFunction;
}
const std::string ObjectiveFunctionId = "ObjectiveFunction";
const bool RegisteredObjectiveFunction = registerCreationFunction<ObjectiveFunction, std::string>(ObjectiveFunctionId, CreateObjectiveFunction);
};

#endif //OBJECTIVEFUNCTION_CUH
