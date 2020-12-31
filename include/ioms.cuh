#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io
{
public:
IoMS();
IoMS(char *original_FITS_name, char *fits_path);
headerValues IoreadCanvas(char *canvas_name);
void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding);
void IocopyMS(char const *infile, char const *outfile);
void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag);
void IocloseCanvas(fitsfile *canvas);
void IoPrintImage(float *I, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU);
void IoPrintImage(float *I, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU);
void IoPrintImageIteration(float *I, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU);
void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, int iteration, bool isInGPU);
void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU);
void doOrderIterations(float *I);
void doOrderEnd(float *I);
void doOrderError(float *I);
};

#endif
