#include <algorithm>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <signal.h>
#include "MSFITSIO.cuh"

#define FLOAT_IMG   -32
#define DOUBLE_IMG  -64

#define TSTRING      16
#define TLONG        41
#define TINT         31
#define TFLOAT       42
#define TDOUBLE      82
#define TCOMPLEX     83
#define TDBLCOMPLEX 163


const float PI = CUDART_PI_F;
const double PI_D = CUDART_PI;
const float RPDEG = (PI/180.0);
const double RPDEG_D = (PI_D/180.0);
const float RPARCM = (PI/(180.0*60.0));
const float LIGHTSPEED = 2.99792458E8;
const float RZ = 1.2196698912665045;

typedef struct varsPerGPU {
        float *device_chi2;
        float *device_dchi2;
        cufftHandle plan;
        cufftComplex *device_Inu;
        cufftComplex *device_V;
}varsPerGPU;

typedef struct variablesPerField {
        float *atten_image;
}VariablesPerField;

typedef struct variables {
        char *input;
        char *output;
        char *inputdat;
        char *modin;
        char *ofile;
        char *path;
        char *output_image;
        char *multigpu;
        char *alpha_name;
        int select;
        int blockSizeX;
        int blockSizeY;
        int blockSizeV;
        int it_max;
        int burndown_steps;
        int reg_term;
        int gridding;
        float noise;
        float noise_cut;
        float randoms;
        float lambda;
        float minpix;
        float nu_0;
        float eta;
        float epsilon;
        float alpha_value;
        float robust_param;
        float noise_factorInu0;
        int current_k;
} Vars;

__host__ void goToError();
__host__ long NearestPowerOf2(long x);
__host__ void readInputDat(char *file);
__host__ void init_beam(int telescope);
__host__ void print_help();
__host__ char *strip(const char *string, const char *chars);
__host__ Vars getOptions(int argc, char **argv);
__host__ float chiCuadrado(float2 *I);
__host__ void dchiCuadrado(float2 *I, float2 *dxi2);
__host__ void do_gridding(Field *fields, freqData *data, double deltau, double deltav, int M, int N, float robust);
__host__ float calculateNoise(Field *fields, freqData data, int *total_visibilities, int blockSizeV, int gridding);
__host__ void clipping(cufftComplex *I, int iterations);
__host__ float deviceReduce(float *in, long N);
__host__ void MetropolisHasting(float2 *I, float2 *theta, int iterations, int *burndown_steps, int *accepted);
__host__ void Metropolis(float2 *I, float2 *theta, int iterations, int burndown_steps, int accepted);

__device__ float EllipticGaussianKernel(float amplitude, int x_c, int y_c, float bmaj, float bmin, float bpa, double DELTAX, double DELTAY);

__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n);
__global__ void clipWNoise(float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta);
__global__ void getGandDGG(float *gg, float *dgg, float2 *xi, float2 *g, long N);
__global__ void newP(float2 *p, float2 *xi, float xmin, long N, float minpix, float fg_scale, float eta);
__global__ void newPNoPositivity(float2 *p, float2 *xi, float xmin, long N);
__global__ void getGGandDGG(float *gg, float *dgg, float2 *xi, float2 *g, long N);
__global__ void clip(cufftComplex *I, long N, float MINPIX);
__global__ void clip2IWNoise(float *noise, float2 *I, long N, float noise_cut, float minpix, float fg_scale, float eta);
__global__ void clip2I(float2 *I, long N, float minpix, float fg_scale);
__global__ void hermitianSymmetry(double3 *UVW, cufftComplex *Vo, float freq, int numVisibilities);
__device__ float attenuation(float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY);
__global__ void total_attenuation(float *total_atten, float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY, long N);
__global__ void mean_attenuation(float *total_atten, int channels, long N);
__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N);
__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N);
__global__ void apply_beam(float beam_fwhm, float beam_freq, float beam_cutoff, cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float fg_scale, float freq, double DELTAX, double DELTAY);
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs);
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, double3 *UVW, float deltau, float deltav, long numVisibilities, long N);
__global__ void alphaVectors(float *alpha_num, float *alpha_den, float *w, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void residual_XCORR(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, float alpha, long numVisibilities);
__global__ void makePositive(cufftComplex *I, long N);
__global__ void evaluateXt(float2 *xt, float2 *pcom, float2 *xicom, float x, long N, float minpix, float fg_scale, float eta);
__global__ void evaluateXtNoPositivity(float2 *xt, float2 *pcom, float2 *xicom, float x, long N);
__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, int numVisibilities);
__global__ void SVector(float *S, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta);
__global__ void QVector(float *Q, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void TVVector(float *TV, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void searchDirection(float2 *g, float2 *xi, float2 *h, long N);
__global__ void newXi(float2 *g, float2 *xi, float2 *h, float gam, long N);
__global__ void clip(cufftComplex *I, float *grad, float RMS, long N);
__global__ void restartDPhi(float2 *dChi2, float *dS, long N);
__global__ void projection(float *px, float *x, float MINPIX, long N);
__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N);
__global__ void normVectorCalculation(float *normVector, float *gc, long N);
__global__ void copyImage(cufftComplex *p, float *device_xt, long N);
__global__ void calculateInu(cufftComplex *I_nu, float2 *image2, float nu, float nu_0, float fg_scale, float minpix, float eta, long N);
__global__ void random_init(unsigned int seed, curandState_t* states, long N);
__global__ void changeGibbsEllipticalGaussianSpecIdx(float2 *temp, float2 *theta, float normal_I_nu_0, float normal_alpha, double crpix1, double crpix2, float bmaj, float bmin, float bpa, float factor, int2 pix, double DELTAX, double DELTAY, int N);
__global__ void changeGibbsEllipticalGaussian(float2 *temp, float2 *theta, float normal_I_nu_0, double crpix1, double crpix2, float bmaj, float bmin, float bpa, float factor, int2 pix, double DELTAX, double DELTAY, int N);
__global__ void changeGibbsEllipticalMaskAlpha(float2 *temp, float2 *theta, float *mask, float normal_I_nu_0, float normal_alpha, double crpix1, double crpix2, float bmaj, float bmin, float bpa, float factor_beam, float factor_noise, float sigma, int2 pix, float DELTAX, float DELTAY, int N);
