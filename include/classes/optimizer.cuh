
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

class Optimizer
{
public:
__host__ virtual void allocateMemoryGpu() = 0;
__host__ virtual void deallocateMemoryGpu() = 0;
__host__ virtual void optimize() = 0;
__host__ virtual int getK(){};
__host__ virtual void setK(int K){};
//__host__ virtual void configure() = 0;

__host__ Optimizer(){
        this->tolerance = 1E-12;
        this->total_iterations = 500;
};

__host__ Optimizer(int total_iterations, float tolerance){
        this->tolerance = tolerance;
        this->total_iterations = total_iterations;
};

__host__ float getTolerance(){
        return this->tolerance;
};

__host__ int getCurrentIteration(){
    return this->current_iteration;
};

__host__ void setImage(Image *image){
        this->image = image;
};
__host__ void setObjectiveFunction(ObjectiveFunction *of){
        this->of = of;
};
void setFlag(int flag){
        this->flag = flag;
};

void setTolerance(float tol){
    this->tolerance = tol;
};

void setTotalIterations(int iterations){
    this->total_iterations = iterations;
};

ObjectiveFunction* getObjectiveFunction(){
        return this->of;
};
protected:
ObjectiveFunction *of;
Image *image;
int flag;
int total_iterations;
int current_iteration = 0;
float tolerance;
};

#endif //OPTIMIZER_CUH
