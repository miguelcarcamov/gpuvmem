#ifndef MSFITSIO_CUH
#define MSFITSIO_CUH

#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/Slicer.h>
#include <casa/Arrays/Vector.h>
#include <cuda.h>
#include <cufft.h>
#include <fitsio.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math_constants.h>
#include <ms/MeasurementSets.h>
#include <ms/MeasurementSets/MSAntennaColumns.h>
#include <ms/MeasurementSets/MSMainColumns.h>
#include <stdio.h>
#include <stdlib.h>
#include <tables/TaQL/TableParse.h>
#include <tables/Tables/ArrColDesc.h>
#include <tables/Tables/ArrayColumn.h>
#include <tables/Tables/ColumnDesc.h>
#include <tables/Tables/ScaColDesc.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableDesc.h>
#include <tables/Tables/TableIter.h>
#include <tables/Tables/TableRow.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum_kahan.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#include <string>

#include "complexOps.cuh"
#include "rngs.cuh"
#include "rvgs.cuh"

#define FLOAT_IMG -32
#define DOUBLE_IMG -64

#define TSTRING 16
#define TLONG 41
#define TINT 31
#define TFLOAT 42
#define TDOUBLE 82
#define TCOMPLEX 83
#define TDBLCOMPLEX 163

const float PI = CUDART_PI_F;
const double PI_D = CUDART_PI;
const float LIGHTSPEED = 2.99792458E8;

enum { AIRYDISK, GAUSSIAN };

typedef struct MSData {
  int n_internal_frequencies;
  std::vector<int> n_internal_frequencies_ids;
  int total_frequencies;
  std::vector<int> channels;
  int nfields;
  int nsamples;
  int nstokes;
  int nantennas;
  int nbaselines;
  float ref_freq;
  float min_freq;
  float max_freq;
  float max_blength;
  float min_blength;
  double uvmax_wavelength;
  std::string telescope_name;
  std::vector<int> corr_type;

  int max_number_visibilities_in_channel_and_stokes;
  int max_number_vis;
} MSData;

typedef struct host_visibilities {
  std::vector<double3> uvw;
  std::vector<float> weight;
  std::vector<cufftComplex> Vo;
  std::vector<cufftComplex> Vm;
  std::vector<cufftComplex> Vr;
  std::vector<int> S;
} HVis;

typedef struct device_visibilities {
  double3* uvw;
  float* weight;
  cufftComplex* Vo;
  cufftComplex* Vm;
  cufftComplex* Vr;
  int* S;

  int threadsPerBlockUV;
  int numBlocksUV;
} DVis;

typedef struct field {
  int id;
  int valid_frequencies;
  double ref_ra, ref_dec;
  double phs_ra, phs_dec;
  float ref_xobs_cartesian, ref_yobs_cartesian;
  float phs_xobs_cartesian, phs_yobs_cartesian;
  float ref_xobs_pix, ref_yobs_pix;
  float phs_xobs_pix, phs_yobs_pix;
  float* atten_image;
  std::vector<float> nu;
  std::vector<std::vector<long>> numVisibilitiesPerFreqPerStoke;
  std::vector<long> numVisibilitiesPerFreq;
  std::vector<std::vector<long>> backup_numVisibilitiesPerFreqPerStoke;
  std::vector<long> backup_numVisibilitiesPerFreq;
  std::vector<std::vector<HVis>> visibilities;
  std::vector<std::vector<DVis>> device_visibilities;
  std::vector<std::vector<HVis>> backup_visibilities;
} Field;

typedef struct MSAntenna {
  std::string antenna_id;
  std::string station;
  double3 position;
  float antenna_diameter;
  float pb_factor;
  float pb_cutoff;
  int primary_beam;
} MSAntenna;

typedef struct MSDataset {
  char* name;
  char* oname;
  std::vector<Field> fields;
  std::vector<MSAntenna> antennas;
  MSData data;
} MSDataset;

typedef struct header_values {
  double DELTAX, DELTAY;
  double ra, dec;
  double crpix1, crpix2;
  long M, N;
  double beam_bmaj, beam_bmin, beam_bpa;
  float beam_noise = -1.0f;
  std::string radesys;
  float equinox;
  int bitpix;
} headerValues;

__host__ headerValues readOpenedFITSHeader(fitsfile*& hdu_in, bool close_fits);
__host__ headerValues readFITSHeader(const char* filename);
__host__ fitsfile* openFITS(const char* filename);
__host__ void closeFITS(fitsfile* canvas);

template <typename T>
__host__ T readHeaderKeyword(char* filename, char* keyword, int type) {
  int status_header = 0;

  T value;

  fitsfile* hdu_in = openFITS(filename);

  fits_read_key(hdu_in, type, keyword, &value, NULL, &status_header);

  if (status_header) {
    fits_report_error(stderr, status_header); /* print error message */
  }

  closeFITS(hdu_in);

  return value;
}

template <typename T>
__host__ headerValues open_fits(T** data, const char* filename, int datatype) {
  int status = 0;
  float null = 0;
  long fpixel = 1;
  int anynull;
  headerValues h_values;

  fitsfile* hdu = openFITS(filename);

  h_values = readOpenedFITSHeader(hdu, false);
  int elements = h_values.M * h_values.N;

  *data = (T*)malloc(elements * sizeof(T));

  fits_read_img(hdu, datatype, fpixel, elements, &null, *data, &anynull,
                &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(0);
  }

  closeFITS(hdu);
  return h_values;
}

__host__ void readMS(const char* MS_name,
                     std::vector<MSAntenna>& antennas,
                     std::vector<Field>& fields,
                     MSData* data,
                     bool noise,
                     bool W_projection,
                     float random_prob,
                     int gridding);
__host__ void readMS(const char* MS_name,
                     std::string data_column,
                     std::vector<MSAntenna>& antennas,
                     std::vector<Field>& fields,
                     MSData* data,
                     bool noise,
                     bool W_projection,
                     float random_prob,
                     int gridding);
__host__ void MScopy(const char* in_dir, const char* in_dir_dest);

__host__ void modelToHost(std::vector<Field>& fields,
                          MSData data,
                          int num_gpus,
                          int firstgpu);
__host__ void writeMS(const char* outfile,
                      const char* out_col,
                      std::vector<Field> fields,
                      MSData data,
                      float random_probability,
                      bool sim,
                      bool noise,
                      bool W_projection);
__host__ void OCopyFITS(float* I,
                        const char* original_filename,
                        const char* path,
                        const char* name_image,
                        char* units,
                        int iteration,
                        int index,
                        float fg_scale,
                        long M,
                        long N,
                        double ra_center,
                        double dec_center,
                        std::string frame,
                        float equinox,
                        bool isInGPU);
__host__ void OCopyFITSCufftComplex(cufftComplex* I,
                                    const char* original_filename,
                                    const char* path,
                                    const char* out_image,
                                    int iteration,
                                    float fg_scale,
                                    long M,
                                    long N,
                                    int option,
                                    bool isInGPU);
__host__ fitsfile* createFITS(const char* filename);
__host__ void copyHeader(fitsfile* original, fitsfile* output);

__host__ __device__ float freq_to_wavelength(float freq);
__host__ __device__ double metres_to_lambda(double uvw_metres, float freq);
__host__ __device__ float distance(float x, float y, float x0, float y0);

#endif
