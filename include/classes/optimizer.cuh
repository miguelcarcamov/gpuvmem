
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

class Optimizer
{
public:
__host__ virtual void allocateMemoryGpu() = 0;
__host__ virtual void deallocateMemoryGpu() = 0;
__host__ virtual void optimize() = 0;
//__host__ virtual void configure() = 0;

__host__ Optimizer::Optimizer(){
        this->tolerance = 1E-12;
        this->total_iterations = 500;
        this->current_iteration = 0;
};

__host__ Optimizer::Optimizer(int total_iterations, float tolerance){
        this->tolerance = tolerance;
        this->total_iterations = total_iterations;
        this->current_iteration = 0;
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

ObjectiveFunction* getObjectiveFuntion(){
        return this->of;
};
protected:
ObjectiveFunction *of;
Image *image;
int flag;
int total_iterations;
int current_iteration;
float tolerance;
};

#endif //OPTIMIZER_CUH
