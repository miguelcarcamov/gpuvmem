
#ifndef OPTIMIZATOR_CUH
#define OPTIMIZATOR_CUH

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

#endif //OPTIMIZATOR_CUH
