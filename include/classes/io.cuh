#ifndef IO_CUH
#define IO_CUH

#include "MSFITSIO.cuh"

class Io
{
public:
virtual headerValues IoreadFITSHeader(char *fits_filename) = 0;
virtual void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding) = 0;
virtual void IocopyMS(char const *infile, char const *outfile) = 0;
virtual void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag) = 0;
virtual void IocloseFITS(fitsfile *hdu) = 0;
virtual void IoPrintImage(float *I, char *original_name, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintImageIteration(float *I, char *original_name, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, bool isInGPU) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, char *original_name, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) = 0;

Io(){
        this->input_name = "";
};

Io(std::string input_name){
        this->input_name = input_name;
};

private:
std::string input_name;
};

#endif
