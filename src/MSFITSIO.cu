/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */

#include "MSFITSIO.cuh"

__host__ __device__ float freq_to_wavelength(float freq) {
  float lambda = LIGHTSPEED / freq;
  return lambda;
}

__host__ __device__ double metres_to_lambda(double uvw_metres, float freq) {
  float lambda = freq_to_wavelength(freq);
  double uvw_lambda = uvw_metres / lambda;
  return uvw_lambda;
}

__host__ __device__ float distance(float x, float y, float x0, float y0) {
  float sumsqr = (x - x0) * (x - x0) + (y - y0) * (y - y0);
  float distance = sqrtf(sumsqr);
  return distance;
}

__host__ fitsfile* openFITS(const char* filename) {
  fitsfile* hdu;
  int status = 0;

  fits_open_file(&hdu, filename, 0, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(0);
  }
  return hdu;
}

__host__ fitsfile* createFITS(const char* filename) {
  fitsfile* fpointer;
  int status = 0;
  fits_create_file(&fpointer, filename, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(-1);
  }
  return fpointer;
}

__host__ void copyHeader(fitsfile* original, fitsfile* output) {
  int status = 0;
  fits_copy_header(original, output, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(-1);
  }
}

__host__ void closeFITS(fitsfile* canvas) {
  int status = 0;
  fits_close_file(canvas, &status);
  if (status) {
    fits_report_error(stderr, status);
    exit(-1);
  }
}

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
                        bool isInGPU) {
  int status = 0;
  long fpixel = 1;
  long elements = M * N;
  size_t needed;
  long naxes[2] = {M, N};
  long naxis = 2;
  char* full_name;

  needed = snprintf(NULL, 0, "!%s%s", path, name_image) + 1;
  full_name = (char*)malloc(needed * sizeof(char));
  snprintf(full_name, needed * sizeof(char), "!%s%s", path, name_image);

  fitsfile* fpointer = createFITS(full_name);
  fitsfile* original_hdu = openFITS(original_filename);
  copyHeader(original_hdu, fpointer);

  fits_update_key(fpointer, TSTRING, "BUNIT", units, "Unit of measurement",
                  &status);
  fits_update_key(fpointer, TINT, "NITER", &iteration,
                  "Number of iteration in gpuvmem software", &status);
  fits_update_key(fpointer, TINT, "NAXIS1", &M, "", &status);
  fits_update_key(fpointer, TINT, "NAXIS2", &N, "", &status);
  fits_update_key(fpointer, TSTRING, "RADESYS", (void*)frame.c_str(),
                  "Changed by gpuvmem", &status);
  fits_update_key(fpointer, TFLOAT, "EQUINOX", &equinox, "Changed by gpuvmem",
                  &status);
  fits_update_key(fpointer, TDOUBLE, "CRVAL1", &ra_center, "Changed by gpuvmem",
                  &status);
  fits_update_key(fpointer, TDOUBLE, "CRVAL2", &dec_center,
                  "Changed by gpuvmem", &status);

  float* host_IFITS = (float*)malloc(M * N * sizeof(float));

  // unsigned int offset = M*N*index*sizeof(float);
  int offset = M * N * index;

  if (isInGPU) {
    checkCudaErrors(cudaMemcpy(host_IFITS, &I[offset], sizeof(float) * M * N,
                               cudaMemcpyDeviceToHost));
  } else {
    memcpy(host_IFITS, &I[offset], M * N * sizeof(float));
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      host_IFITS[N * i + j] *= fg_scale;
    }
  }

  fits_write_img(fpointer, TFLOAT, fpixel, elements, host_IFITS, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(-1);
  }

  closeFITS(original_hdu);
  closeFITS(fpointer);

  free(host_IFITS);
}

__host__ void OCopyFITSCufftComplex(cufftComplex* I,
                                    const char* original_filename,
                                    const char* path,
                                    const char* out_image,
                                    int iteration,
                                    float fg_scale,
                                    long M,
                                    long N,
                                    int option,
                                    bool isInGPU) {
  int status = 0;
  long fpixel = 1;
  long elements = M * N;
  size_t needed;
  char* name;
  long naxes[2] = {M, N};
  long naxis = 2;
  char* unit = "JY/PIXEL";

  switch (option) {
    case 0:
      needed = snprintf(NULL, 0, "!%s", out_image) + 1;
      name = (char*)malloc(needed * sizeof(char));
      snprintf(name, needed * sizeof(char), "!%s", out_image);
      break;
    case 1:
      needed = snprintf(NULL, 0, "!%sMEM_%d.fits", path, iteration) + 1;
      name = (char*)malloc(needed * sizeof(char));
      snprintf(name, needed * sizeof(char), "!%sMEM_%d.fits", path, iteration);
      break;
    case -1:
      break;
    default:
      printf("Invalid case to FITS\n");
      exit(-1);
  }

  fitsfile* fpointer = createFITS(name);
  fitsfile* original_hdu = openFITS(original_filename);
  copyHeader(original_hdu, fpointer);

  fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement",
                  &status);
  fits_update_key(fpointer, TINT, "NITER", &iteration,
                  "Number of iteration in gpuvmem software", &status);

  cufftComplex* host_IFITS;
  host_IFITS = (cufftComplex*)malloc(M * N * sizeof(cufftComplex));
  float* image2D = (float*)malloc(M * N * sizeof(float));
  if (isInGPU) {
    checkCudaErrors(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I,
                                 sizeof(cufftComplex), sizeof(cufftComplex),
                                 M * N, cudaMemcpyDeviceToHost));
  } else {
    memcpy(host_IFITS, I, M * N * sizeof(cufftComplex));
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      /*Amplitude*/
      image2D[N * i + j] = amplitude(host_IFITS[N * i + j]);
      /* Phase in degrees */
      // image2D[N*i+j] = phaseDegrees<cufftComplex, float>(host_IFITS[N*i+j]);
      /*Real part*/
      // image2D[N*i+j] = host_IFITS[N*i+j].x;
      /*Imaginary part*/
      // image2D[N*i+j] = host_IFITS[N*i+j].y;
    }
  }

  fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(-1);
  }

  closeFITS(original_hdu);
  closeFITS(fpointer);

  free(host_IFITS);
  free(image2D);
  free(name);
}

__host__ headerValues readOpenedFITSHeader(fitsfile*& hdu_in, bool close_fits) {
  int status_header = 0;
  int status_noise = 0;
  int status_dirty_beam = 0;
  int status_radesys = 0;
  int status_equinox = 0;
  float aux_noise;

  headerValues h_values;
  int bitpix;
  char* aux_radesys;
  int radesys_length;

  fits_get_key_strlen(hdu_in, "RADESYS", &radesys_length, &status_header);

  aux_radesys = (char*)malloc(radesys_length * sizeof(char));

  fits_read_key(hdu_in, TDOUBLE, "CDELT1", &h_values.DELTAX, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CDELT2", &h_values.DELTAY, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRVAL1", &h_values.ra, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRVAL2", &h_values.dec, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRPIX1", &h_values.crpix1, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRPIX2", &h_values.crpix2, NULL,
                &status_header);
  fits_read_key(hdu_in, TLONG, "NAXIS1", &h_values.M, NULL, &status_header);
  fits_read_key(hdu_in, TLONG, "NAXIS2", &h_values.N, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "BMAJ", &h_values.beam_bmaj, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TDOUBLE, "BMIN", &h_values.beam_bmin, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TDOUBLE, "BPA", &h_values.beam_bpa, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TFLOAT, "NOISE", &aux_noise, NULL, &status_noise);
  fits_read_key(hdu_in, TSTRING, "RADESYS", aux_radesys, NULL, &status_radesys);
  fits_read_key(hdu_in, TFLOAT, "EQUINOX", &h_values.equinox, NULL,
                &status_equinox);

  h_values.radesys = aux_radesys;

  fits_get_img_type(hdu_in, &bitpix, &status_header);
  h_values.bitpix = bitpix;

  if (status_header) {
    fits_report_error(stderr, status_header); /* print error message */
    exit(0);
  }

  if (!status_noise) {
    h_values.beam_noise = aux_noise;
  }

  if (status_equinox) {
    h_values.equinox = 2000.0;
  }

  if (close_fits)
    closeFITS(hdu_in);

  return h_values;
}

__host__ headerValues readFITSHeader(const char* filename) {
  int status_header = 0;
  int status_noise = 0;
  int status_radesys = 0;
  int status_equinox = 0;
  int status_dirty_beam = 0;
  float aux_noise;

  headerValues h_values;
  int bitpix;
  char* aux_radesys;
  int radesys_length;

  fitsfile* hdu_in = openFITS(filename);

  fits_get_key_strlen(hdu_in, "RADESYS", &radesys_length, &status_header);

  aux_radesys = (char*)malloc(radesys_length * sizeof(char));

  fits_read_key(hdu_in, TDOUBLE, "CDELT1", &h_values.DELTAX, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CDELT2", &h_values.DELTAY, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRVAL1", &h_values.ra, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRVAL2", &h_values.dec, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRPIX1", &h_values.crpix1, NULL,
                &status_header);
  fits_read_key(hdu_in, TDOUBLE, "CRPIX2", &h_values.crpix2, NULL,
                &status_header);
  fits_read_key(hdu_in, TLONG, "NAXIS1", &h_values.M, NULL, &status_header);
  fits_read_key(hdu_in, TLONG, "NAXIS2", &h_values.N, NULL, &status_header);
  fits_read_key(hdu_in, TDOUBLE, "BMAJ", &h_values.beam_bmaj, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TDOUBLE, "BMIN", &h_values.beam_bmin, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TDOUBLE, "BPA", &h_values.beam_bpa, NULL,
                &status_dirty_beam);
  fits_read_key(hdu_in, TFLOAT, "NOISE", &aux_noise, NULL, &status_noise);
  fits_read_key(hdu_in, TSTRING, "RADESYS", aux_radesys, NULL, &status_radesys);
  fits_read_key(hdu_in, TFLOAT, "EQUINOX", &h_values.equinox, NULL,
                &status_equinox);

  h_values.radesys = aux_radesys;

  fits_get_img_type(hdu_in, &bitpix, &status_header);
  h_values.bitpix = bitpix;

  if (status_header) {
    fits_report_error(stderr, status_header); /* print error message */
    exit(0);
  }

  if (!status_noise) {
    h_values.beam_noise = aux_noise;
  }

  if (status_equinox) {
    h_values.equinox = 2000.0;
  }

  closeFITS(hdu_in);
  return h_values;
}

__host__ cufftComplex addNoiseToVis(cufftComplex vis, float weights) {
  cufftComplex noise_vis;

  float real_n = Normal(0, 1);
  float imag_n = Normal(0, 1);

  noise_vis = make_cuFloatComplex(vis.x + real_n * (1 / sqrtf(weights)),
                                  vis.y + imag_n * (1 / sqrtf(weights)));

  return noise_vis;
}

constexpr unsigned int str2int(const char* str, int h = 0) {
  return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}


__host__ void MScopy(const char* in_dir, const char* in_dir_dest) {
  string dir_origin = in_dir;
  string dir_dest = in_dir_dest;

  casacore::Table tab_src(dir_origin);
  tab_src.deepCopy(dir_dest, casacore::Table::New);
}
