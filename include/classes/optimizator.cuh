
#ifndef OPTIMIZATOR_CUH
#define OPTIMIZATOR_CUH

class Optimizator
{
public:
__host__ virtual void allocateMemoryGpu() = 0;
__host__ virtual void deallocateMemoryGpu() = 0;
__host__ virtual void optimize() = 0;
//__host__ virtual void configure() = 0;

__host__ Optimizator::Optimizator(){
        this->tolerance = 1E-12;
}

__host__ float getTolerance(){
        return this->tolerance;
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

ObjectiveFunction* getObjectiveFuntion(){
        return this->of;
};
protected:
ObjectiveFunction *of;
Image *image;
int flag;
float tolerance;
};

#endif //OPTIMIZATOR_CUH
