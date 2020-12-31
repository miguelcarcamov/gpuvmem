#ifndef IO_CUH
#define IO_CUH

#include "MSFITSIO.cuh"

class Io
{
public:

virtual void setFitsPath(char *fits_path)
{
        this->fits_path = fits_path;
};
virtual void setOriginal_FITS_name(char *original_FITS_name)
{
        this->original_FITS_name = original_FITS_name;
};
virtual headerValues IoreadCanvas(char *canvas_name) = 0;
virtual void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding) = 0;
virtual void IocopyMS(char const *infile, char const *outfile) = 0;
virtual void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag) = 0;
virtual void IocloseCanvas(fitsfile *canvas) = 0;
virtual void IoPrintImage(float *I, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintImage(float *I, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintImageIteration(float *I, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, int iteration, bool isInGPU) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) = 0;
void setPrintImagesPath(char * pip){
        this->printImagesPath = pip;
};
protected:
int *iteration;
char *printImagesPath;
char *original_FITS_name;
char *fits_path;
};

#endif
