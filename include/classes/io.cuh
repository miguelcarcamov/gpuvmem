#ifndef IO_CUH
#define IO_CUH

#include "MSFITSIO.cuh"

class Io
{
public:
virtual canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag) = 0;
virtual void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding) = 0;
virtual void IocopyMS(char const *infile, char const *outfile) = 0;
virtual void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag) = 0;
virtual void IocloseCanvas(fitsfile *canvas) = 0;
virtual void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, bool isInGPU) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU) = 0;
void setPrintImagesPath(char * pip){
        this->printImagesPath = pip;
};
protected:
int *iteration;
char *printImagesPath;
};

#endif
