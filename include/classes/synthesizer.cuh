#ifndef GPUVMEM_SYNTHESIZER_CUH
#define GPUVMEM_SYNTHESIZER_CUH
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

    __host__ void setWeightingScheme(WeightingScheme *scheme){
        this->scheme = scheme;
    };
    __host__ void setOrder(void (*func)(Optimizator *o, Image *I)){
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
    WeightingScheme *scheme = NULL;
};
#endif //GPUVMEM_SYNTHESIZER_CUH
