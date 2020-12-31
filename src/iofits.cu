#include "iofits.cuh"


IoFITS::IoFITS() : Io()
{
        this->conversion_factor = 1.0f;
        this->path = "";
};

IoFITS::IoFITS(std::string input_name, float conversion_factor, std::string path) : Io(input_name)
{
        this->conversion_factor = conversion_factor;
        this->path = path;
};

headerValues IoFITS::IoreadFITSHeader(char *fits_filename)
{
        return readFITSHeader(fits_filename);
};

void IoFITS::IoPrintImage(float *I, char *original_name, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU)
{
        OCopyFITS(I, original_name, path, name_image, units, iteration, index, fg_scale, M, N, isInGPU);
};

void IoFITS::IoPrintImageIteration(float *I, char *original_name, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU)
{
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iteration);

        std::string unit_str(units);

        if(unit_str == "JY/PIXEL")
                OCopyFITS(I, original_name, path, full_name, units, iteration, index, fg_scale, M, N, isInGPU);
        else
                OCopyFITS(I, original_name, path, full_name, units, iteration, index, 1.0, M, N, isInGPU);

        free(full_name);
};


void IoFITS::IoPrintcuFFTComplex(cufftComplex *I, char *original_name, char *out_image, char *path, int iteration, float fg_scale, long M, long N, int option, bool isInGPU)
{
        OCopyFITSCufftComplex(I, original_name, out_image, path, iteration, fg_scale, M, N, option, isInGPU);
};

void IoFITS::IocloseFITS(fitsfile *hdu)
{
        closeFITS(hdu);
};

float IoFITS::getConversionFactor(){
        return this->conversion_factor;
};

void IoFITS::setConversionFactor(float conversion_factor){
        this->conversion_factor = conversion_factor;
};
std::string IoFITS::getPath(){
        return this->path;
};

void IoFITS::setPath(std::string path){
        this->path = path;
};

namespace {
Io* CreateIoFITS()
{
        return new IoFITS;
}
const std::string IoFITSId = "IoFITS";
const bool RegisteredIoFITS = registerCreationFunction<Io, std::string>(IoFITSId, CreateIoFITS);
};
