#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io
{
public:
canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag);
void IoreadMS(char const *MS_name, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding);
void IocopyMS(char const *infile, char const *outfile);
void IowriteMS(char const *infile, char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag);
void IocloseCanvas(fitsfile *canvas);
void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N);
void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N);
void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index);
void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option);
void doOrderIterations(float *I);
void doOrderEnd(float *I);
void doOrderError(float *I);
};

#endif
