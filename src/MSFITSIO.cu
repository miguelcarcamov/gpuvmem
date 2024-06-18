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

__host__ void readMS(const char* MS_name,
                     std::vector<MSAntenna>& antennas,
                     std::vector<Field>& fields,
                     MSData* data,
                     bool noise,
                     bool W_projection,
                     float random_prob,
                     int gridding) {
  char* error = 0;
  int g = 0, h = 0;

  std::string dir(MS_name);
  casacore::Table main_tab(dir);
  std::string data_column;

  data->nsamples = main_tab.nrow();
  if (data->nsamples == 0) {
    printf("ERROR: nsamples is zero... exiting....\n");
    exit(-1);
  }

  if (main_tab.tableDesc().isColumn("CORRECTED_DATA") &&
      main_tab.tableDesc().isColumn("DATA"))
    data_column = "CORRECTED_DATA";
  else if (main_tab.tableDesc().isColumn("CORRECTED_DATA"))
    data_column = "CORRECTED_DATA";
  else if (main_tab.tableDesc().isColumn("DATA"))
    data_column = "DATA";
  else {
    printf(
        "ERROR: There is no column CORRECTED_DATA OR DATA in this Measurement "
        "SET. Exiting...\n");
    exit(-1);
  }

  printf("GPUVMEM is reading %s data column\n", data_column.c_str());

  casacore::Vector<double> pointing_ref;
  casacore::Vector<double> pointing_phs;
  int pointing_id;

  casacore::Table observation_tab(main_tab.keywordSet().asTable("OBSERVATION"));
  casacore::ROScalarColumn<casacore::String> obs_col(observation_tab,
                                                     "TELESCOPE_NAME");

  data->telescope_name = obs_col(0);
  std::string ref_dir = "REFERENCE_DIR";
  std::string phase_dir = "PHASE_DIR";
  std::string field_query = "select " + ref_dir + "," + phase_dir +
                            ",ROWID() AS ID FROM " + dir +
                            "/FIELD where !FLAG_ROW";
  casacore::Table field_tab(casacore::tableCommand(field_query.c_str()));

  std::string spw_query =
      "select NUM_CHAN,CHAN_FREQ FROM " + dir + "/SPECTRAL_WINDOW t1 JOIN " +
      dir +
      "/DATA_DESCRIPTION t2 ON t1.rownumber()=t2.SPECTRAL_WINDOW_ID where "
      "!FLAG_ROW";
  casacore::Table spectral_window_tab(
      casacore::tableCommand(spw_query.c_str()));

  std::cout << "Spectral window table has " << spectral_window_tab.nrow()
            << " rows";

  std::string pol_query = "select NUM_CORR,CORR_TYPE FROM " + dir +
                          "/POLARIZATION t1 JOIN " + dir +
                          "/DATA_DESCRIPTION t2 ON "
                          "t1.rownumber()=t2.POLARIZATION_ID where !FLAG_ROW";
  casacore::Table polarization_tab(casacore::tableCommand(pol_query.c_str()));
  std::cout << "Polarization table has " << polarization_tab.nrow() << " rows";

  std::string antenna_tab_query =
      "select POSITION,DISH_DIAMETER,NAME,STATION FROM " + dir +
      "/ANTENNA where !FLAG_ROW";
  std::string maxmin_baseline_query =
      "select GMAX(B_LENGTH) AS MAX_BLENGTH, GMIN(B_LENGTH) AS MIN_BLENGTH \
        FROM (select sqrt(sumsqr(UVW[:2])) as B_LENGTH FROM " +
      dir + " GROUPBY ANTENNA1,ANTENNA2)";
  std::string freq_query =
      "select GMIN(CHAN_FREQ) as MIN_FREQ, GMAX(CHAN_FREQ) as MAX_FREQ, "
      "GMEDIAN(CHAN_FREQ) as REF_FREQ FROM " +
      dir + "/SPECTRAL_WINDOW";
  std::string maxuv_metres_query =
      "select MAX(GMAX(UVW[0]),GMAX(UVW[1])) as MAXUV FROM " + dir;

  casacore::Table antenna_tab(
      casacore::tableCommand(antenna_tab_query.c_str()));
  casacore::Table maxmin_baseline_tab(
      casacore::tableCommand(maxmin_baseline_query.c_str()));
  casacore::Table freq_tab(casacore::tableCommand(freq_query.c_str()));
  casacore::Table maxuv_metres_tab(
      casacore::tableCommand(maxuv_metres_query.c_str()));

  casacore::ROScalarColumn<casacore::Double> max_blength_col(
      maxmin_baseline_tab, "MAX_BLENGTH");
  casacore::ROScalarColumn<casacore::Double> min_blength_col(
      maxmin_baseline_tab, "MIN_BLENGTH");
  casacore::ROScalarColumn<casacore::Double> maxuv_metres_col(maxuv_metres_tab,
                                                              "MAXUV");

  casacore::ROScalarColumn<casacore::Double> min_freq_col(freq_tab, "MIN_FREQ");
  casacore::ROScalarColumn<casacore::Double> max_freq_col(freq_tab, "MAX_FREQ");
  casacore::ROScalarColumn<casacore::Double> ref_freq_col(freq_tab, "REF_FREQ");

  data->nantennas = antenna_tab.nrow();
  data->nbaselines = (data->nantennas) * (data->nantennas - 1) / 2;
  data->ref_freq = ref_freq_col(0);
  data->min_freq = min_freq_col(0);
  data->max_freq = max_freq_col(0);
  data->max_blength = max_blength_col(0);
  data->min_blength = min_blength_col(0);
  data->uvmax_wavelength = maxuv_metres_col(0) * data->max_freq / LIGHTSPEED;

  float max_wavelength = freq_to_wavelength(data->min_freq);

  casacore::ROArrayColumn<casacore::Double> dishposition_col(antenna_tab,
                                                             "POSITION");
  casacore::ROScalarColumn<casacore::Double> dishdiameter_col(antenna_tab,
                                                              "DISH_DIAMETER");
  casacore::ROScalarColumn<casacore::String> dishname_col(antenna_tab, "NAME");
  casacore::ROScalarColumn<casacore::String> dishstation_col(antenna_tab,
                                                             "STATION");

  casacore::Vector<double> antenna_positions;

  float firstj1zero = boost::math::cyl_bessel_j_zero(1.0f, 1);
  float pb_defaultfactor = firstj1zero / PI;

  for (int a = 0; a < data->nantennas; a++) {
    antennas.push_back(MSAntenna());
    antennas[a].antenna_id = dishname_col(a);
    antennas[a].station = dishstation_col(a);
    antenna_positions = dishposition_col(a);
    antennas[a].position.x = antenna_positions[0];
    antennas[a].position.y = antenna_positions[1];
    antennas[a].position.z = antenna_positions[2];
    antennas[a].antenna_diameter = dishdiameter_col(a);

    switch (str2int((data->telescope_name).c_str())) {
      case str2int("ALMA"):
        antennas[a].pb_factor = 1.13f;
        antennas[a].primary_beam = AIRYDISK;
        break;
      case str2int("EVLA"):
        antennas[a].pb_factor = 1.25f;
        antennas[a].primary_beam = GAUSSIAN;
        break;
      default:
        antennas[a].pb_factor = pb_defaultfactor;
        antennas[a].primary_beam = GAUSSIAN;
        break;
    }

    antennas[a].pb_cutoff = 10.0f * antennas[a].pb_factor *
                            (max_wavelength / antennas[a].antenna_diameter);
  }

  data->nfields = field_tab.nrow();
  casacore::ROTableRow field_row(
      field_tab, casacore::stringToVector("ID,REFERENCE_DIR,PHASE_DIR"));

  for (int f = 0; f < data->nfields; f++) {
    const casacore::TableRecord& values = field_row.get(f);
    pointing_id = values.asInt("ID");
    pointing_ref = values.asArrayDouble("REFERENCE_DIR");
    pointing_phs = values.asArrayDouble("PHASE_DIR");

    fields.push_back(Field());

    fields[f].id = pointing_id;
    fields[f].ref_ra = pointing_ref[0];
    fields[f].ref_dec = pointing_ref[1];

    fields[f].phs_ra = pointing_phs[0];
    fields[f].phs_dec = pointing_phs[1];
  }

  casacore::ROScalarColumn<casacore::Int64> ncorr_col(polarization_tab,
                                                      "NUM_CORR");
  data->nstokes = ncorr_col(0);
  casacore::ROArrayColumn<casacore::Int64> correlation_col(polarization_tab,
                                                           "CORR_TYPE");
  casacore::Vector<casacore::Int64> polarizations = correlation_col(0);

  for (int i = 0; i < data->nstokes; i++) {
    data->corr_type.push_back(polarizations[i]);
  }

  data->n_internal_frequencies = spectral_window_tab.nrow();

  casacore::ROArrayColumn<casacore::Double> chan_freq_col(spectral_window_tab,
                                                          "CHAN_FREQ");

  casacore::ROScalarColumn<casacore::Int64> n_chan_freq(spectral_window_tab,
                                                        "NUM_CHAN");

  casacore::ROScalarColumn<casacore::Int64> spectral_window_ids(
      spectral_window_tab, "ID");

  for (int i = 0; i < data->n_internal_frequencies; i++) {
    data->n_internal_frequencies_ids.push_back(spectral_window_ids(i));
    data->channels.push_back(n_chan_freq(i));
  }

  int total_frequencies = 0;
  for (int i = 0; i < data->n_internal_frequencies; i++) {
    for (int j = 0; j < data->channels[i]; j++) {
      total_frequencies++;
    }
  }

  data->total_frequencies = total_frequencies;

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      casacore::Vector<double> chan_freq_vector;
      chan_freq_vector = chan_freq_col(i);
      for (int j = 0; j < data->channels[i]; j++) {
        fields[f].nu.push_back(chan_freq_vector[j]);
      }
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    fields[f].visibilities.resize(data->total_frequencies,
                                  std::vector<HVis>(data->nstokes, HVis()));
    fields[f].device_visibilities.resize(
        data->total_frequencies, std::vector<DVis>(data->nstokes, DVis()));
    fields[f].numVisibilitiesPerFreqPerStoke.resize(
        data->total_frequencies, std::vector<long>(data->nstokes, 0));
    fields[f].numVisibilitiesPerFreq.resize(data->total_frequencies, 0);
    fields[f].backup_visibilities.resize(
        data->total_frequencies, std::vector<HVis>(data->nstokes, HVis()));
    if (gridding) {
      fields[f].backup_numVisibilitiesPerFreqPerStoke.resize(
          data->total_frequencies, std::vector<long>(data->nstokes, 0));
      fields[f].backup_numVisibilitiesPerFreq.resize(data->total_frequencies,
                                                     0);
    }
  }

  std::string query;

  casacore::Vector<float> weights;
  casacore::Vector<double> uvw;
  casacore::Matrix<casacore::Complex> dataCol;
  casacore::Matrix<bool> flagCol;

  double3 MS_uvw;
  cufftComplex MS_vis;
  for (int f = 0; f < data->nfields; f++) {
    g = 0;
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      dataCol.resize(data->nstokes, data->channels[i]);
      flagCol.resize(data->nstokes, data->channels[i]);

      query = "select UVW,WEIGHT," + data_column + ",FLAG from " + dir +
              " where DATA_DESC_ID in [select from ::DATA_DESCRIPTION where "
              "SPECTRAL_WINDOW_ID=" +
              std::to_string(data->n_internal_frequencies_ids[i]) +
              " and !FLAG_ROW giving [ROWID()]] and FIELD_ID=" +
              std::to_string(fields[f].id) + " and !FLAG_ROW and ANY(!FLAG)";
      if (W_projection && random_prob < 1.0) {
        query += " and RAND()<" + std::to_string(random_prob) +
                 " ORDERBY ASC UVW[2]";
      } else if (W_projection) {
        query += " ORDERBY ASC UVW[2]";
      } else if (random_prob < 1.0) {
        query += " and RAND()<" + std::to_string(random_prob);
      }

      casacore::Table query_tab = casacore::tableCommand(query.c_str());

      casacore::ROArrayColumn<double> uvw_col(query_tab, "UVW");
      casacore::ROArrayColumn<float> weight_col(query_tab, "WEIGHT");
      casacore::ROArrayColumn<casacore::Complex> data_col(query_tab,
                                                          data_column);
      casacore::ROArrayColumn<bool> flag_col(query_tab, "FLAG");

      for (int k = 0; k < query_tab.nrow(); k++) {
        uvw = uvw_col(k);
        dataCol = data_col(k);
        weights = weight_col(k);
        flagCol = flag_col(k);
        for (int j = 0; j < data->channels[i]; j++) {
          for (int sto = 0; sto < data->nstokes; sto++) {
            if (weights[sto] > 0.0f && flagCol(sto, j) == false) {
              MS_uvw.x = uvw[0];
              MS_uvw.y = uvw[1];
              MS_uvw.z = uvw[2];

              fields[f].visibilities[g + j][sto].uvw.push_back(MS_uvw);

              MS_vis = make_cuFloatComplex(dataCol(sto, j).real(),
                                           dataCol(sto, j).imag());

              if (noise)
                fields[f].visibilities[g + j][sto].Vo.push_back(
                    addNoiseToVis(MS_vis, weights[sto]));
              else
                fields[f].visibilities[g + j][sto].Vo.push_back(MS_vis);

              fields[f].visibilities[g + j][sto].weight.push_back(weights[sto]);
              fields[f].numVisibilitiesPerFreqPerStoke[g + j][sto]++;
              fields[f].numVisibilitiesPerFreq[g + j]++;
            }
          }
        }
      }
      g += data->channels[i];
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      for (int sto = 0; sto < data->nstokes; sto++) {
        fields[f].numVisibilitiesPerFreq[i] +=
            fields[f].numVisibilitiesPerFreqPerStoke[i][sto];
        /*
         *
         * We will allocate memory for model visibilities using the size of the
         * observed visibilities vector.
         */
        fields[f].visibilities[i][sto].Vm.assign(
            fields[f].visibilities[i][sto].Vo.size(),
            make_cuFloatComplex(0.0f, 0.0f));
      }
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    h = 0;
    fields[f].valid_frequencies = 0;
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      for (int j = 0; j < data->channels[i]; j++) {
        if (fields[f].numVisibilitiesPerFreq[h] > 0) {
          fields[f].valid_frequencies++;
        }
        h++;
      }
    }
  }

  int local_max = 0;
  int max = 0;
  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      local_max = *std::max_element(
          fields[f].numVisibilitiesPerFreqPerStoke[i].data(),
          fields[f].numVisibilitiesPerFreqPerStoke[i].data() + data->nstokes);
      if (local_max > max) {
        max = local_max;
      }
    }
  }

  data->max_number_visibilities_in_channel_and_stokes = max;
}

__host__ void readMS(const char* MS_name,
                     std::string data_column,
                     std::vector<MSAntenna>& antennas,
                     std::vector<Field>& fields,
                     MSData* data,
                     bool noise,
                     bool W_projection,
                     float random_prob,
                     int gridding) {
  char* error = 0;
  int g = 0, h = 0;

  std::string dir(MS_name);
  casacore::Table main_tab(dir);

  data->nsamples = main_tab.nrow();
  if (data->nsamples == 0) {
    printf("ERROR: nsamples is zero... exiting....\n");
    exit(-1);
  }

  if (!main_tab.tableDesc().isColumn(data_column)) {
    printf(
        "ERROR: There is no column CORRECTED_DATA OR DATA in this Measurement "
        "SET. Exiting...\n");
    exit(-1);
  }

  printf("GPUVMEM is reading %s data column\n", data_column.c_str());

  casacore::Vector<double> pointing_ref;
  casacore::Vector<double> pointing_phs;
  int pointing_id;

  casacore::Table observation_tab(main_tab.keywordSet().asTable("OBSERVATION"));
  casacore::ROScalarColumn<casacore::String> obs_col(observation_tab,
                                                     "TELESCOPE_NAME");

  data->telescope_name = obs_col(0);

  std::string field_query =
      "select REFERENCE_DIR,PHASE_DIR,ROWID() AS ID FROM " + dir +
      "/FIELD where !FLAG_ROW";
  casacore::Table field_tab(casacore::tableCommand(field_query.c_str()));

  std::string aux_query = "select DATA_DESC_ID FROM " + dir +
                          " WHERE !FLAG_ROW AND ANY(WEIGHT > 0) AND ANY(!FLAG) "
                          "ORDER BY UNIQUE DATA_DESC_ID";
  std::string spw_query =
      "select NUM_CHAN,CHAN_FREQ,ROWID() AS ID FROM " + dir +
      "/SPECTRAL_WINDOW where !FLAG_ROW AND ANY(ROWID()==[" + aux_query + "])";
  casacore::Table spectral_window_tab(
      casacore::tableCommand(spw_query.c_str()));

  std::string pol_query = "select NUM_CORR,CORR_TYPE,ROWID() AS ID FROM " +
                          dir + "/POLARIZATION where !FLAG_ROW";
  casacore::Table polarization_tab(casacore::tableCommand(pol_query.c_str()));

  std::string antenna_tab_query =
      "select POSITION,DISH_DIAMETER,NAME,STATION FROM " + dir +
      "/ANTENNA where !FLAG_ROW";
  std::string maxmin_baseline_query =
      "select GMAX(B_LENGTH) AS MAX_BLENGTH, GMIN(B_LENGTH) AS MIN_BLENGTH \
        FROM (select sqrt(sumsqr(UVW[:2])) as B_LENGTH FROM " +
      dir + " GROUPBY ANTENNA1,ANTENNA2)";
  std::string freq_query =
      "select GMIN(CHAN_FREQ) as MIN_FREQ, GMAX(CHAN_FREQ) as MAX_FREQ, "
      "GMEDIAN(CHAN_FREQ) as REF_FREQ FROM " +
      dir + "/SPECTRAL_WINDOW";
  std::string maxuv_metres_query =
      "select MAX(GMAX(UVW[0]),GMAX(UVW[1])) as MAXUV FROM " + dir;

  casacore::Table antenna_tab(
      casacore::tableCommand(antenna_tab_query.c_str()));
  casacore::Table maxmin_baseline_tab(
      casacore::tableCommand(maxmin_baseline_query.c_str()));
  casacore::Table freq_tab(casacore::tableCommand(freq_query.c_str()));
  casacore::Table maxuv_metres_tab(
      casacore::tableCommand(maxuv_metres_query.c_str()));

  casacore::ROScalarColumn<casacore::Double> max_blength_col(
      maxmin_baseline_tab, "MAX_BLENGTH");
  casacore::ROScalarColumn<casacore::Double> min_blength_col(
      maxmin_baseline_tab, "MIN_BLENGTH");
  casacore::ROScalarColumn<casacore::Double> maxuv_metres_col(maxuv_metres_tab,
                                                              "MAXUV");

  casacore::ROScalarColumn<casacore::Double> min_freq_col(freq_tab, "MIN_FREQ");
  casacore::ROScalarColumn<casacore::Double> max_freq_col(freq_tab, "MAX_FREQ");
  casacore::ROScalarColumn<casacore::Double> ref_freq_col(freq_tab, "REF_FREQ");

  data->nantennas = antenna_tab.nrow();
  data->nbaselines = (data->nantennas) * (data->nantennas - 1) / 2;
  data->ref_freq = ref_freq_col(0);
  data->min_freq = min_freq_col(0);
  data->max_freq = max_freq_col(0);
  data->max_blength = max_blength_col(0);
  data->min_blength = min_blength_col(0);
  data->uvmax_wavelength = maxuv_metres_col(0) * data->max_freq / LIGHTSPEED;

  float max_wavelength = freq_to_wavelength(data->min_freq);

  casacore::ROArrayColumn<casacore::Double> dishposition_col(antenna_tab,
                                                             "POSITION");
  casacore::ROScalarColumn<casacore::Double> dishdiameter_col(antenna_tab,
                                                              "DISH_DIAMETER");
  casacore::ROScalarColumn<casacore::String> dishname_col(antenna_tab, "NAME");
  casacore::ROScalarColumn<casacore::String> dishstation_col(antenna_tab,
                                                             "STATION");

  casacore::Vector<double> antenna_positions;

  float firstj1zero = boost::math::cyl_bessel_j_zero(1.0f, 1);
  float pb_defaultfactor = firstj1zero / PI;

  for (int a = 0; a < data->nantennas; a++) {
    antennas.push_back(MSAntenna());
    antennas[a].antenna_id = dishname_col(a);
    antennas[a].station = dishstation_col(a);
    antenna_positions = dishposition_col(a);
    antennas[a].position.x = antenna_positions[0];
    antennas[a].position.y = antenna_positions[1];
    antennas[a].position.z = antenna_positions[2];
    antennas[a].antenna_diameter = dishdiameter_col(a);

    switch (str2int((data->telescope_name).c_str())) {
      case str2int("ALMA"):
        antennas[a].pb_factor = 1.13f;
        antennas[a].primary_beam = AIRYDISK;
        break;
      case str2int("EVLA"):
        antennas[a].pb_factor = 1.25f;
        antennas[a].primary_beam = GAUSSIAN;
        break;
      default:
        antennas[a].pb_factor = pb_defaultfactor;
        antennas[a].primary_beam = GAUSSIAN;
        break;
    }

    antennas[a].pb_cutoff = 10.0f * antennas[a].pb_factor *
                            (max_wavelength / antennas[a].antenna_diameter);
  }

  data->nfields = field_tab.nrow();
  casacore::ROTableRow field_row(
      field_tab, casacore::stringToVector("ID,REFERENCE_DIR,PHASE_DIR"));

  for (int f = 0; f < data->nfields; f++) {
    const casacore::TableRecord& values = field_row.get(f);
    pointing_id = values.asInt("ID");
    pointing_ref = values.asArrayDouble("REFERENCE_DIR");
    pointing_phs = values.asArrayDouble("PHASE_DIR");

    fields.push_back(Field());

    fields[f].id = pointing_id;
    fields[f].ref_ra = pointing_ref[0];
    fields[f].ref_dec = pointing_ref[1];

    fields[f].phs_ra = pointing_phs[0];
    fields[f].phs_dec = pointing_phs[1];
  }

  casacore::ROScalarColumn<casacore::Int64> ncorr_col(polarization_tab,
                                                      "NUM_CORR");
  data->nstokes = ncorr_col(0);
  casacore::ROArrayColumn<casacore::Int64> correlation_col(polarization_tab,
                                                           "CORR_TYPE");
  casacore::Vector<casacore::Int64> polarizations = correlation_col(0);

  for (int i = 0; i < data->nstokes; i++) {
    data->corr_type.push_back(polarizations[i]);
  }

  data->n_internal_frequencies = spectral_window_tab.nrow();

  casacore::ROArrayColumn<casacore::Double> chan_freq_col(spectral_window_tab,
                                                          "CHAN_FREQ");

  casacore::ROScalarColumn<casacore::Int64> n_chan_freq(spectral_window_tab,
                                                        "NUM_CHAN");

  casacore::ROScalarColumn<casacore::Int64> spectral_window_ids(
      spectral_window_tab, "ID");

  for (int i = 0; i < data->n_internal_frequencies; i++) {
    data->n_internal_frequencies_ids.push_back(spectral_window_ids(i));
    data->channels.push_back(n_chan_freq(i));
  }

  int total_frequencies = 0;
  for (int i = 0; i < data->n_internal_frequencies; i++) {
    for (int j = 0; j < data->channels[i]; j++) {
      total_frequencies++;
    }
  }

  data->total_frequencies = total_frequencies;

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      casacore::Vector<double> chan_freq_vector;
      chan_freq_vector = chan_freq_col(i);
      for (int j = 0; j < data->channels[i]; j++) {
        fields[f].nu.push_back(chan_freq_vector[j]);
      }
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    fields[f].visibilities.resize(data->total_frequencies,
                                  std::vector<HVis>(data->nstokes, HVis()));
    fields[f].device_visibilities.resize(
        data->total_frequencies, std::vector<DVis>(data->nstokes, DVis()));
    fields[f].numVisibilitiesPerFreqPerStoke.resize(
        data->total_frequencies, std::vector<long>(data->nstokes, 0));
    fields[f].numVisibilitiesPerFreq.resize(data->total_frequencies, 0);
    fields[f].backup_visibilities.resize(
        data->total_frequencies, std::vector<HVis>(data->nstokes, HVis()));
    if (gridding) {
      fields[f].backup_numVisibilitiesPerFreqPerStoke.resize(
          data->total_frequencies, std::vector<long>(data->nstokes, 0));
      fields[f].backup_numVisibilitiesPerFreq.resize(data->total_frequencies,
                                                     0);
    }
  }

  std::string query;

  casacore::Vector<float> weights;
  casacore::Vector<double> uvw;
  casacore::Matrix<casacore::Complex> dataCol;
  casacore::Matrix<bool> flagCol;

  double3 MS_uvw;
  cufftComplex MS_vis;
  for (int f = 0; f < data->nfields; f++) {
    g = 0;
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      dataCol.resize(data->nstokes, data->channels[i]);
      flagCol.resize(data->nstokes, data->channels[i]);

      query = "select UVW,WEIGHT," + data_column + ",FLAG from " + dir +
              " where DATA_DESC_ID in [select from ::DATA_DESCRIPTION where "
              "SPECTRAL_WINDOW_ID=" +
              std::to_string(data->n_internal_frequencies_ids[i]) +
              " and !FLAG_ROW giving [ROWID()]] and FIELD_ID=" +
              std::to_string(fields[f].id) + " and !FLAG_ROW and ANY(!FLAG)";
      if (W_projection && random_prob < 1.0) {
        query += " and RAND()<" + std::to_string(random_prob) +
                 " ORDERBY ASC UVW[2]";
      } else if (W_projection) {
        query += " ORDERBY ASC UVW[2]";
      } else if (random_prob < 1.0) {
        query += " and RAND()<" + std::to_string(random_prob);
      }

      casacore::Table query_tab = casacore::tableCommand(query.c_str());

      casacore::ROArrayColumn<double> uvw_col(query_tab, "UVW");
      casacore::ROArrayColumn<float> weight_col(query_tab, "WEIGHT");
      casacore::ROArrayColumn<casacore::Complex> data_col(query_tab,
                                                          data_column);
      casacore::ROArrayColumn<bool> flag_col(query_tab, "FLAG");

      for (int k = 0; k < query_tab.nrow(); k++) {
        uvw = uvw_col(k);
        dataCol = data_col(k);
        weights = weight_col(k);
        flagCol = flag_col(k);
        for (int j = 0; j < data->channels[i]; j++) {
          for (int sto = 0; sto < data->nstokes; sto++) {
            if (weights[sto] > 0.0f && flagCol(sto, j) == false) {
              MS_uvw.x = uvw[0];
              MS_uvw.y = uvw[1];
              MS_uvw.z = uvw[2];

              fields[f].visibilities[g + j][sto].uvw.push_back(MS_uvw);

              MS_vis = make_cuFloatComplex(dataCol(sto, j).real(),
                                           dataCol(sto, j).imag());

              if (noise)
                fields[f].visibilities[g + j][sto].Vo.push_back(
                    addNoiseToVis(MS_vis, weights[sto]));
              else
                fields[f].visibilities[g + j][sto].Vo.push_back(MS_vis);

              fields[f].visibilities[g + j][sto].weight.push_back(weights[sto]);
              fields[f].numVisibilitiesPerFreqPerStoke[g + j][sto]++;
              fields[f].numVisibilitiesPerFreq[g + j]++;
            }
          }
        }
      }
      g += data->channels[i];
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      for (int sto = 0; sto < data->nstokes; sto++) {
        fields[f].numVisibilitiesPerFreq[i] +=
            fields[f].numVisibilitiesPerFreqPerStoke[i][sto];
        /*
         *
         * We will allocate memory for model visibilities using the size of the
         * observed visibilities vector.
         */
        fields[f].visibilities[i][sto].Vm.assign(
            fields[f].visibilities[i][sto].Vo.size(),
            make_cuFloatComplex(0.0f, 0.0f));
      }
    }
  }

  for (int f = 0; f < data->nfields; f++) {
    h = 0;
    fields[f].valid_frequencies = 0;
    for (int i = 0; i < data->n_internal_frequencies; i++) {
      for (int j = 0; j < data->channels[i]; j++) {
        if (fields[f].numVisibilitiesPerFreq[h] > 0) {
          fields[f].valid_frequencies++;
        }
        h++;
      }
    }
  }

  int local_max = 0;
  int max = 0;
  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      local_max = *std::max_element(
          fields[f].numVisibilitiesPerFreqPerStoke[i].data(),
          fields[f].numVisibilitiesPerFreqPerStoke[i].data() + data->nstokes);
      if (local_max > max) {
        max = local_max;
      }
    }
  }

  data->max_number_visibilities_in_channel_and_stokes = max;
}

__host__ void MScopy(const char* in_dir, const char* in_dir_dest) {
  string dir_origin = in_dir;
  string dir_dest = in_dir_dest;

  casacore::Table tab_src(dir_origin);
  tab_src.deepCopy(dir_dest, casacore::Table::New);
}

__host__ void modelToHost(std::vector<Field>& fields,
                          MSData data,
                          int num_gpus,
                          int firstgpu) {
  for (int f = 0; f < data.nfields; f++) {
    for (int i = 0; i < data.total_frequencies; i++) {
      cudaSetDevice((i % num_gpus) + firstgpu);
      for (int s = 0; s < data.nstokes; s++) {
        checkCudaErrors(
            cudaMemcpy(fields[f].visibilities[i][s].Vm.data(),
                       fields[f].device_visibilities[i][s].Vm,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                       cudaMemcpyDeviceToHost));
        for (int j = 0; j < fields[f].numVisibilitiesPerFreqPerStoke[i][s];
             j++) {
          if (fields[f].visibilities[i][s].uvw[j].x > 0) {
            fields[f].visibilities[i][s].Vm[j] =
                cuConjf(fields[f].visibilities[i][s].Vm[j]);
          }
        }
      }
    }
  }
}

__host__ void writeMS(const char* outfile,
                      const char* out_col,
                      std::vector<Field> fields,
                      MSData data,
                      float random_probability,
                      bool sim,
                      bool noise,
                      bool W_projection) {
  std::string dir = outfile;
  casacore::Table main_tab(dir, casacore::Table::Update);
  std::string column_name(out_col);
  std::string query;

  if (main_tab.tableDesc().isColumn(column_name)) {
    printf("Column %s already exists... skipping creation...\n", out_col);
  } else {
    printf("Adding %s to the main table...\n", out_col);
    main_tab.addColumn(casacore::ArrayColumnDesc<casacore::Complex>(
        column_name, "created by gpuvmem"));
    query = "UPDATE " + dir + " SET " + column_name + "=DATA";
    // query = "COPY COLUMN DATA TO MODEL";
    printf("Duplicating DATA column into %s ...\n", column_name.c_str());
    casacore::tableCommand(query.c_str());
    main_tab.flush();
  }

  for (int f = 0; f < data.nfields; f++) {
    for (auto& i : fields[f].numVisibilitiesPerFreqPerStoke)
      std::fill(i.begin(), i.end(), 0);
  }

  int g = 0;
  long c;
  cufftComplex vis;
  SelectStream(0);
  PutSeed(-1);

  casacore::Vector<float> weights;
  casacore::Matrix<casacore::Complex> dataCol;
  casacore::Matrix<bool> flagCol;

  for (int f = 0; f < data.nfields; f++) {
    g = 0;
    for (int i = 0; i < data.n_internal_frequencies; i++) {
      dataCol.resize(data.nstokes, data.channels[i]);
      flagCol.resize(data.nstokes, data.channels[i]);

      query = "select WEIGHT," + column_name + ",FLAG from " + dir +
              " where DATA_DESC_ID in [select from ::DATA_DESCRIPTION where "
              "SPECTRAL_WINDOW_ID=" +
              std::to_string(data.n_internal_frequencies_ids[i]) +
              " and !FLAG_ROW giving [ROWID()]] and FIELD_ID=" +
              std::to_string(fields[f].id) + " and !FLAG_ROW and ANY(!FLAG)";
      if (W_projection)
        query += " ORDERBY ASC UVW[2]";

      casacore::Table query_tab = casacore::tableCommand(query.c_str());

      casacore::ArrayColumn<float> weight_col(query_tab, "WEIGHT");
      casacore::ArrayColumn<casacore::Complex> data_col(query_tab, column_name);
      casacore::ArrayColumn<bool> flag_col(query_tab, "FLAG");

      for (int k = 0; k < query_tab.nrow(); k++) {
        weights = weight_col(k);
        dataCol = data_col(k);
        flagCol = flag_col(k);
        for (int j = 0; j < data.channels[i]; j++) {
          for (int sto = 0; sto < data.nstokes; sto++) {
            if (flagCol(sto, j) == false && weights[sto] > 0.0f) {
              c = fields[f].numVisibilitiesPerFreqPerStoke[g + j][sto];

              if (sim && noise) {
                vis = addNoiseToVis(fields[f].visibilities[g + j][sto].Vm[c],
                                    weights[sto]);
              } else if (sim) {
                vis = fields[f].visibilities[g + j][sto].Vm[c];
              } else {
                vis = cuCsubf(fields[f].visibilities[g + j][sto].Vo[c],
                              fields[f].visibilities[g + j][sto].Vm[c]);
              }

              dataCol(sto, j) = casacore::Complex(vis.x, vis.y);
              weights[sto] = fields[f].visibilities[g + j][sto].weight[c];
              fields[f].numVisibilitiesPerFreqPerStoke[g + j][sto]++;
            }
          }
        }
        data_col.put(k, dataCol);
        weight_col.put(k, weights);
      }

      query_tab.flush();

      string sub_query = "select from " + dir +
                         " where DATA_DESC_ID in [select from "
                         "::DATA_DESCRIPTION where SPECTRAL_WINDOW_ID=" +
                         std::to_string(data.n_internal_frequencies_ids[i]) +
                         " and !FLAG_ROW giving [ROWID()]] and FIELD_ID=" +
                         std::to_string(fields[f].id) +
                         " and !FLAG_ROW and ANY(!FLAG)";
      if (W_projection)
        sub_query += " ORDERBY ASC UVW[2]";

      query = "update [" + sub_query + "], $1 tq set " + column_name +
              "[!FLAG]=tq." + column_name + "[!tq.FLAG], WEIGHT=tq.WEIGHT";

      casacore::TaQLResult result1 =
          casacore::tableCommand(query.c_str(), query_tab);

      g += data.channels[i];
    }
  }

  main_tab.flush();
}
