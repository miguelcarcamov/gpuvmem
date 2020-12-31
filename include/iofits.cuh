#ifndef IOFITS_CUH
#define IOFITS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoFITS : public Io
{
public:
IoFITS();
IoFITS(std::string input_name, float conversion_factor, std::string path);
headerValues IoreadFITSHeader(char *fits_filename);

template <typename T>
headerValues IoreadFITS(std::vector<T> &data)
{
        headerValues hdu = open_fits<T>(&data.data(), this->input_name);
        return hdu;
};
void IoPrintImage(float *I, char *original_name, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU);
void IoPrintImageIteration(float *I, char *original_name, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU);
void IoPrintcuFFTComplex(cufftComplex *I, char *original_name, char *path, char *out_image, int iteration, float fg_scale, long M, long N, int option, bool isInGPU);
void IocloseFITS(fitsfile *hdu);
void doOrderIterations(float *I);
void doOrderEnd(float *I);
void doOrderError(float *I);
float getConversionFactor();
void setConversionFactor(float conversion_factor);
std::string getPath();
void setPath(std::string path);
private:
float conversion_factor;
std::string path;
};

#endif
