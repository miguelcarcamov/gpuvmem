#ifndef IOFITS_CUH
#define IOFITS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoFITS : public Io
{
public:

IoFITS();
IoFITS(std::string input, std::string output, std::string path);
IoFITS(std::string input, std::string output, std::string path, int M, int N, float normalization_factor, bool print_images);
bool getPrintImages() override;
void setM(int M) override;
void setN(int N) override;
void setMN(int M, int N) override;
void setNormalizationFactor(int normalization_factor) override;
void setPrintImages(bool print_images) override;
headerValues readHeader(char *header_name) override;
headerValues readHeader(std::string header_name) override;
headerValues readHeader() override;
void printImage(float *I, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printImage(float *I, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printImage(float *I, char *name_image, char *units, int iteration, int index, bool isInGPU) override;
void printImage(float *I, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printNotPathImage(float *I, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printNotPathImage(float *I, char *out_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printNotPathImage(float *I, char *out_image, char *units, int iteration, int index, bool isInGPU) override;
void printNotPathImage(float *I, char *out_image, char *units, int iteration, int index, float normalization_factor, bool isInGPU) override;
void printNotPathImage(float *I, char *units, int iteration, int index, float normalization_factor, bool isInGPU) override;
void printNotNormalizedImage(float *I, char *name_image, char *units, int iteration, int index, bool isInGPU) override;
void printNotPathNotNormalizedImage(float *I, char *name_image, char *units, int iteration, int index, bool isInGPU) override;
void printImageIteration(float *I, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printImageIteration(float *I, char const *name_image, char *units, int iteration, int index, bool isInGPU) override;
void printNotNormalizedImageIteration(float *I, char const *name_image, char *units, int iteration, int index, bool isInGPU) override;
void printImageIteration(float *I, char *model_input, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) override;
void printcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) override;
void printcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, int option, bool isInGPU) override;
void printcuFFTComplex(cufftComplex *I, char *input, char*path, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) override;
void closeHeader(fitsfile *header) override;

protected:
  int M;
  int N;
  float normalization_factor;
  bool print_images;
};

#endif
