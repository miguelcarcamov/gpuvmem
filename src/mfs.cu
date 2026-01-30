#include "imageProcessor.cuh"
#include "gridder.cuh"
#include "mfs.cuh"
#include "ms/data_column.h"
#include "ms/ms_reader.h"
#include "ms/ms_writer.h"
#include "ms/polarization.h"
#include "objective_function/terms/chi2/chi2.cuh"
#include "objective_function/terms/regularizers/secondderivateerror.cuh"
#include "functions.cuh"

#include <algorithm>

long M, N, numVisibilities;

float *device_Image, *device_dphi, *device_dchi2_total, *device_dS, *device_S,
    *device_noise_image, *device_weight_image, *device_distance_image;
float noise_cut, MINPIX, minpix, random_probability = 1.0;
float noise_jypix, eta, robust_param;
float *host_I, sum_weights, *penalizators;
double beam_bmaj, beam_bmin, beam_bpa;

// Global Image pointer for backward compatibility (used as fallback in line searchers)
// Line searchers prefer using this->image from optimizer, but fall back to extern I if needed
Image* I;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int status_mod_in;
int multigpu, firstgpu, reg_term, total_visibilities, image_count,
    nPenalizators, nMeasurementSets = 0, max_number_vis;

std::string msinput, msoutput, modinput, mempath, out_image, output;
float nu_0, threshold, alpha_n_sigma;
extern int num_gpus;

double ra, dec, crpix1, crpix2, DELTAX, DELTAY, deltau, deltav;

std::string radesys;

float equinox;

std::vector<float> initial_values;
std::vector<gpuvmem::ms::MSWithGPU> datasets;
std::vector<gpuvmem::ms::MSWithGPU>* g_datasets = nullptr;

varsPerGPU* vars_gpu;

bool verbose_flag, nopositivity, apply_noise, print_images, print_errors,
    save_model_input, radius_mask, modify_weights;

Vars variables;

clock_t t;
double start, end;

float noise_min = 1E32;

Flags flags;

inline bool IsGPUCapableP2P(cudaDeviceProp* pProp) {
#ifdef _WIN32
  return (bool)(pProp->tccDriver ? true : false);
#else
  return (bool)(pProp->major >= 2);
#endif
}

// Helper function to get memory clock rate compatible with all CUDA versions
// CUDA_VERSION format: (MAJOR * 1000 + MINOR * 10), e.g., CUDA 13.0 = 13000
inline int GetMemoryClockRateKHz(int deviceId, cudaDeviceProp* pProp) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
  // CUDA 13.0+: clockRate and memoryClockRate removed from struct, use
  // attribute API
  int memoryClockRateKHz = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&memoryClockRateKHz,
                                         cudaDevAttrMemoryClockRate, deviceId));
  return memoryClockRateKHz;
#else
  // CUDA < 13.0: Use struct member (more efficient, direct access)
  return pProp->memoryClockRate;
#endif
}

std::vector<std::string> MFS::countAndSeparateStrings(std::string long_str,
                                                      std::string sep) {
  std::vector<std::string> ret;
  boost::split(ret, long_str, boost::is_any_of(sep));

  return ret;
}

void MFS::configure(int argc, char** argv) {
  if (ioImageHandler == NULL) {
    ioImageHandler = createObject<Io, std::string>("IoFITS");
  }

  if (ioVisibilitiesHandler == NULL) {
    ioVisibilitiesHandler = createObject<Io, std::string>("IoMS");
  }

  total_visibilities = 0;
  variables = getOptions(argc, argv);
  msinput = variables.input;
  msoutput = variables.output;
  modinput = variables.modin;
  ioImageHandler->setInput(modinput);
  out_image = variables.output_image;
  ioImageHandler->setOutput(out_image);
  ioImageHandler->setPath(variables.path);
  optimizer->setTotalIterations(variables.it_max);
  this->setVisNoise(variables.noise);
  noise_cut = variables.noise_cut;
  random_probability = variables.randoms;
  ioVisibilitiesHandler->setRandomProbability(random_probability);
  eta = variables.eta;
  // Ensure eta is negative for proper positivity clipping
  // eta defaults to -1.0f, but if user provides a non-negative value, use default
  if (eta >= 0.0f) {
    if (verbose_flag) {
      std::cerr << "WARNING: eta must be negative for positivity clipping. Using default eta = -1.0" << std::endl;
    }
    eta = -1.0f;
  }
  ioVisibilitiesHandler->setGridding(variables.gridding);
  this->setGriddingThreads(variables.gridding);
  nu_0 = variables.nu_0;
  robust_param = variables.robust_param;
  threshold = variables.threshold * 5.0;
  alpha_n_sigma = variables.alpha_n_sigma;
  ioVisibilitiesHandler->setApplyNoiseInput(apply_noise);
  ioVisibilitiesHandler->setStoreModelVisInput(save_model_input);
  ioImageHandler->setPrintImages(print_images);
  this->ckernel->setIoImageHandler(ioImageHandler);

  std::vector<std::string> string_values;
  std::vector<std::string> s_output_values;
  int n_outputs;

  if (msinput != "NULL") {
    string_values = countAndSeparateStrings(msinput, ",");
    nMeasurementSets = string_values.size();
  } else {
    printf("Datasets files were not provided\n");
    print_help();
    exit(-1);
  }

  if (msoutput != "NULL") {
    s_output_values = countAndSeparateStrings(msoutput, ",");
    n_outputs = s_output_values.size();
  } else {
    printf("Output/s was/were not provided\n");
    print_help();
    exit(-1);
  }

  if (n_outputs != nMeasurementSets) {
    printf(
        "Number of input datasets should be equal to the number of output "
        "datasets\n");
    exit(-1);
  }

  if (verbose_flag)
    printf("Number of input datasets %d\n", nMeasurementSets);

  for (int i = 0; i < nMeasurementSets; i++) {
    gpuvmem::ms::MSWithGPU dw;
    dw.name = string_values[i];
    dw.oname = s_output_values[i];
    datasets.push_back(std::move(dw));
  }

  string_values.clear();
  s_output_values.clear();

  std::vector<std::string> requested_stokes;
  if (variables.initial_values != "NULL") {
    string_values = countAndSeparateStrings(variables.initial_values, ",");
    image_count = string_values.size();
  } else {
    printf("Initial values for image/s were not provided\n");
    print_help();
    exit(-1);
  }

  // Store minimal_pixel_values from initial_values (before eta multiplication)
  // These will be stored in Image object
  // Store as member variable so it's accessible in setDevice() where Image object is created
  this->minimal_pixel_values.clear();
  for (int i = 0; i < image_count; i++) {
    if (i == 0) {
      // Set MINPIX from the first initial value (before eta multiplication)
      // This is the minimum pixel value threshold for positivity clipping
      MINPIX = std::stof(string_values[i]);
      // Store MINPIX * -1.0f * eta in minimal_pixel_values (same as initial_values)
      this->minimal_pixel_values.push_back(MINPIX * -1.0f * eta);
      // Store value after eta multiplication in initial_values
      initial_values.push_back(MINPIX * -1.0f * eta);
    } else {
      this->minimal_pixel_values.push_back(std::stof(string_values[i]));
      initial_values.push_back(std::stof(string_values[i]));
    }
  }

  string_values.clear();
  if (!variables.stokes.empty()) {
    // Stokes imaging: image_count = number of requested Stokes; validate against datasets after read
    requested_stokes = countAndSeparateStrings(variables.stokes, ",");
    image_count = static_cast<int>(requested_stokes.size());
    if (image_count <= 0) {
      printf("ERROR: --stokes must specify at least one Stokes (e.g. I or I,Q,U,V)\n");
      exit(-1);
    }
    // Ensure minimal_pixel_values and initial_values have image_count entries (pad with 0.0f)
    while (static_cast<int>(this->minimal_pixel_values.size()) < image_count)
      this->minimal_pixel_values.push_back(0.0f);
    while (static_cast<int>(initial_values.size()) < image_count)
      initial_values.push_back(0.0f);
  } else if (image_count == 1) {
    initial_values.push_back(0.0f);
    this->minimal_pixel_values.push_back(0.0f);  // Add default minimal pixel value for second image
    image_count++;
    imagesChanged = 1;
  }

  /*
     Read FITS header
   */
  headerValues header_vars = ioImageHandler->readHeader(modinput);
  // canvas_vars.beam_noise =
  // iohandler->readHeaderKeyword<float>(strdup(modinput.c_str()), "NOISE",
  // TFLOAT);
  M = header_vars.M;
  N = header_vars.N;
  ioImageHandler->setMN(M, N);
  DELTAX = header_vars.DELTAX;
  DELTAY = header_vars.DELTAY;
  ra = header_vars.ra;
  dec = header_vars.dec;
  ioImageHandler->setRADec(ra, dec);
  radesys = header_vars.radesys;
  ioImageHandler->setFrame(radesys);
  equinox = header_vars.equinox;
  printf("Equinox %f\n", equinox);
  ioImageHandler->setEquinox(equinox);
  crpix1 = header_vars.crpix1;
  crpix2 = header_vars.crpix2;
  if (header_vars.beam_noise > 0.0f) {
    this->setVisNoise(header_vars.beam_noise);
  }

  ckernel->setIoImageHandler(ioImageHandler);
  // printf("Beam size canvas: %lf x %lf (arcsec)/ %lf (degrees)\n",
  // canvas_vars.beam_bmaj*3600.0, canvas_vars.beam_bmin*3600.0,
  // canvas_vars.beam_bpa);
  cudaGetDeviceCount(&num_gpus);
  cudaDeviceProp dprop[num_gpus];

  printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("Number of CUDA devices:\t%d\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    checkCudaErrors(cudaGetDeviceProperties(&dprop[i], i));
    printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i,
           dprop[i].name, (IsGPUCapableP2P(&dprop[i]) ? "IS " : "NOT"));

    // Get memory clock rate - compatible with all CUDA versions
    int memoryClockRateKHz = GetMemoryClockRateKHz(i, &dprop[i]);
    printf("> Memory Clock Rate (KHz): %d\n", memoryClockRateKHz);
    printf("> Memory Bus Width (bits): %d\n", dprop[i].memoryBusWidth);
    printf("> Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * memoryClockRateKHz * (dprop[i].memoryBusWidth / 8) / 1.0e6);
    printf("> Total Global Memory (GB): %f\n",
           dprop[i].totalGlobalMem / pow(2, 30));
    printf("-----------------------------------------------------------\n");
  }
  printf("-----------------------------------------------------------\n\n");

  // Declaring block size and number of blocks for Image
  if (variables.blockSizeX == -1 && variables.blockSizeY == -1) {
    int maxGridSizeX, maxGridSizeY;
    int numblocksX, numblocksY;
    int threadsX, threadsY;
    maxGridSizeX = iDivUp(M, sqrt(256));
    maxGridSizeY = iDivUp(N, sqrt(256));
    getNumBlocksAndThreads(M, maxGridSizeX, sqrt(256), numblocksX, threadsX,
                           false);
    getNumBlocksAndThreads(N, maxGridSizeY, sqrt(256), numblocksY, threadsY,
                           false);
    numBlocksNN.x = numblocksX;
    numBlocksNN.y = numblocksY;
    threadsPerBlockNN.x = threadsX;
    threadsPerBlockNN.y = threadsY;
    printf("Your 2D grid is [%d,%d] blocks and each has [%d,%d] threads\n",
           numBlocksNN.x, numBlocksNN.y, threadsPerBlockNN.x,
           threadsPerBlockNN.y);
  } else {
    if (variables.blockSizeX * variables.blockSizeY >
            dprop[0].maxThreadsPerBlock ||
        variables.blockSizeV > dprop[0].maxThreadsPerBlock) {
      printf("Block size X: %d\n", variables.blockSizeX);
      printf("Block size Y: %d\n", variables.blockSizeY);
      printf("Block size X*Y: %d\n",
             variables.blockSizeX * variables.blockSizeY);
      printf("Block size V: %d\n", variables.blockSizeV);
      printf("ERROR. The maximum threads per block cannot be greater than %d\n",
             dprop[0].maxThreadsPerBlock);
      exit(-1);
    }

    if (variables.blockSizeX > dprop[0].maxThreadsDim[0] ||
        variables.blockSizeY > dprop[0].maxThreadsDim[1]) {
      printf("Block size X: %d\n", variables.blockSizeX);
      printf("Block size Y: %d\n", variables.blockSizeY);
      printf("Block size V: %d\n", variables.blockSizeV);
      printf(
          "ERROR. The size of the blocksize cannot exceed X: %d Y: %d Z: %d\n",
          dprop[0].maxThreadsDim[0], dprop[0].maxThreadsDim[1],
          dprop[0].maxThreadsDim[2]);
      exit(-1);
    }
    threadsPerBlockNN.x = variables.blockSizeX;
    threadsPerBlockNN.y = variables.blockSizeY;

    numBlocksNN.x = iDivUp(M, threadsPerBlockNN.x);
    numBlocksNN.y = iDivUp(N, threadsPerBlockNN.y);
  }

  if (verbose_flag)
    printf("Reading data from MSs\n");

  std::vector<float> ms_ref_freqs;
  std::vector<float> ms_max_freqs;
  std::vector<float> ms_min_freqs;
  std::vector<float> ms_max_blength;
  std::vector<float> ms_min_blength;
  std::vector<float> ms_uvmax_wavelength;
  auto reader = gpuvmem::ms::create_ms_reader();
  gpuvmem::ms::MSReadOptions read_opts;
  read_opts.random_probability = random_probability;
  for (int d = 0; d < nMeasurementSets; d++) {
    if (!reader->read(datasets[d].name, datasets[d].ms, read_opts)) {
      printf("Failed to read MS: %s\n", datasets[d].name.c_str());
      exit(-1);
    }
    float min_f = 1e30f, max_f = 0.f;
    for (const auto& spw : datasets[d].ms.metadata().spectral_windows()) {
      for (double f : spw.frequencies()) {
        if (f < min_f) min_f = static_cast<float>(f);
        if (f > max_f) max_f = static_cast<float>(f);
      }
    }
    if (min_f > max_f) min_f = max_f;
    ms_ref_freqs.push_back(0.5f * (min_f + max_f));
    ms_max_freqs.push_back(max_f);
    ms_min_freqs.push_back(min_f);
    ms_max_blength.push_back(1e10f);
    ms_min_blength.push_back(0.f);
    ms_uvmax_wavelength.push_back(1e-5f);
    float ant_diam = 0.f;
    if (!datasets[d].ms.metadata().antennas().empty())
      ant_diam = datasets[d].ms.metadata().antennas()[0].antenna_diameter;
    printf("Dataset %d: %s - Antenna diameter: %.3f metres\n", d,
           datasets[d].name.c_str(), ant_diam);
  }

  // Validate that all datasets can form the requested Stokes (when --stokes was set)
  if (!requested_stokes.empty()) {
    for (int d = 0; d < nMeasurementSets; d++) {
      if (!gpuvmem::ms::stokes_supported_by_metadata(datasets[d].ms.metadata(),
                                                      requested_stokes)) {
        gpuvmem::ms::PolarizationHelper helper(&datasets[d].ms.metadata());
        std::vector<std::string> available;
        for (const auto& dd : datasets[d].ms.metadata().data_descriptions())
          for (const std::string& a :
               helper.available_stokes_for_data_desc(dd.data_desc_id()))
            if (std::find(available.begin(), available.end(), a) == available.end())
              available.push_back(a);
        printf("ERROR: dataset %d (%s) cannot form requested Stokes (",
               d, datasets[d].name.c_str());
        for (size_t i = 0; i < requested_stokes.size(); i++)
          printf("%s%s", requested_stokes[i].c_str(),
                 i + 1 < requested_stokes.size() ? "," : "");
        printf("). Available from correlations: ");
        for (size_t i = 0; i < available.size(); i++)
          printf("%s%s", available[i].c_str(),
                 i + 1 < available.size() ? "," : "");
        printf("\n");
        exit(-1);
      }
    }
    if (verbose_flag)
      printf("Stokes imaging: %d plane(s) (%s)\n", image_count,
             variables.stokes.c_str());
    // Convert visibility data to Stokes so pol index = Stokes index (I=0, Q=1, U=2, V=3)
    for (int d = 0; d < nMeasurementSets; d++) {
      if (!gpuvmem::ms::correlations_to_stokes(datasets[d].ms,
                                                requested_stokes)) {
        printf("ERROR: failed to convert dataset %d (%s) to Stokes\n", d,
               datasets[d].name.c_str());
        exit(-1);
      }
    }
  }

  /*
     Calculating theoretical resolution
   */
  float max_freq = *max_element(ms_max_freqs.begin(), ms_max_freqs.end());
  float min_freq = *min_element(ms_min_freqs.begin(), ms_min_freqs.end());
  float max_blength =
      *max_element(ms_max_blength.begin(), ms_max_blength.end());
  float min_wlength = freq_to_wavelength(max_freq);
  float resolution_arcsec = (min_wlength / max_blength) / RPARCSEC;
  double max_uvmax_wavelength =
      *max_element(ms_uvmax_wavelength.begin(), ms_uvmax_wavelength.end()) +
      1E-5;
  printf("The maximum u,v in wavelength units is: %e\n", max_uvmax_wavelength);
  printf(
      "The maximum theoretical resolution of this/these dataset/s is ~%f "
      "arcsec\n",
      resolution_arcsec);
  printf(
      "The oversampled (by a factor of 7) resolution of this/these dataset/s "
      "is ~%f arcsec\n",
      resolution_arcsec / 7.0f);

  if (nu_0 < 0.0) {
    printf(
        "WARNING: Reference frequency not provided. It will be calculated as "
        "the middle"
        " of the frequency range.\n");
    nu_0 = 0.5f * (max_freq + min_freq);
  }
  printf("Reference frequency: %e Hz\n", nu_0);
  // Alpha error depends on log(nu/nu_0). Print range so users can check
  // leverage.
  double log_nu_min =
      log(static_cast<double>(min_freq) / static_cast<double>(nu_0));
  double log_nu_max =
      log(static_cast<double>(max_freq) / static_cast<double>(nu_0));
  printf("Frequency range: [%.5e, %.5e] Hz -> log(nu/nu_0) in [%.4f, %.4f]\n",
         static_cast<double>(min_freq), static_cast<double>(max_freq),
         log_nu_min, log_nu_max);
  if (fabs(log_nu_max - log_nu_min) < 0.01 && image_count > 1) {
    printf(
        "WARNING: log(nu/nu_0) range is very narrow; alpha will be poorly "
        "constrained and alpha errors may be large or capped.\n");
  }
  double deltau_theo = 2.0 * max_uvmax_wavelength / (M - 1);
  double deltax_theo = 1.0 / (M * deltau_theo) / RPARCSEC;
  printf("The pixel size has to be less or equal to %lf arcsec\n", deltax_theo);
  printf("Actual pixel size is %lf arcsec\n", fabs(DELTAX) * 3600.0);

  if (verbose_flag) {
    for (int i = 0; i < nMeasurementSets; i++) {
      size_t nchan = 0;
      int npol = 0;
      for (const auto& dd : datasets[i].ms.metadata().data_descriptions()) {
        nchan += dd.nchan();
        if (dd.npol() > 0) npol = dd.npol();
      }
      printf("Dataset %d: %s\n", i, datasets[i].name.c_str());
      printf("\tNumber of fields = %zu\n", datasets[i].ms.num_fields());
      printf("\tNumber of frequencies = %zu\n", nchan);
      printf("\tNumber of correlations = %d\n", npol);
    }
  }

  multigpu = 0;
  firstgpu = 0;
  int count_gpus;

  string_values = countAndSeparateStrings(variables.gpus, ",");
  count_gpus = string_values.size();

  if (count_gpus == 0) {
    multigpu = 0;
    firstgpu = 0;
  } else if (count_gpus == 1) {
    multigpu = 0;
    firstgpu = std::stoi(string_values[0]);
  } else {
    multigpu = count_gpus;
    firstgpu = std::stoi(string_values[0]);
  }

  string_values.clear();
  this->ckernel->setGPUID(firstgpu);
  if (variables.penalization_factors != "NULL") {
    string_values =
        countAndSeparateStrings(variables.penalization_factors, ",");
    nPenalizators = string_values.size();
    penalizators = (float*)malloc(sizeof(float) * nPenalizators);
    for (int i = 0; i < nPenalizators; i++) {
      penalizators[i] = std::stof(string_values[i]);
    }

  } else {
    printf("No regularization factors provided\n");
  }
  string_values.clear();

  int max_nfreq = 1;
  if (multigpu < 0 || multigpu > num_gpus) {
    printf(
        "ERROR. NUMBER OF GPUS CANNOT BE NEGATIVE OR GREATER THAN THE NUMBER "
        "OF GPUS\n");
    exit(-1);
  } else {
    if (multigpu == 0) {
      num_gpus = 1;
    } else {
      for (int d = 0; d < nMeasurementSets; d++) {
        int nfreq = 0;
        for (const auto& dd : datasets[d].ms.metadata().data_descriptions())
          nfreq += dd.nchan();
        if (nfreq > max_nfreq) max_nfreq = nfreq;
      }

      if (max_nfreq == 1) {
        printf("ONLY ONE FREQUENCY. CHANGING NUMBER OF GPUS TO 1\n");
        num_gpus = 1;
      } else {
        num_gpus = multigpu;
      }
    }
  }

  int total_gpus;
  cudaGetDeviceCount(&total_gpus);
  if (firstgpu > total_gpus - 1 || firstgpu < 0) {
    printf("ERROR. The selected GPU ID does not exist\n");
    exit(-1);
  }

  if (verbose_flag) {
    printf("Number of CUDA devices and threads: %d\n", num_gpus);
  }

  // Check peer access if there is more than 1 GPU
  if (num_gpus > 1) {
    for (int i = firstgpu + 1; i < firstgpu + num_gpus; i++) {
      cudaDeviceProp dprop0, dpropX;
      cudaGetDeviceProperties(&dprop0, firstgpu);
      cudaGetDeviceProperties(&dpropX, i);
      int canAccessPeer0_x, canAccessPeerx_0;
      cudaDeviceCanAccessPeer(&canAccessPeer0_x, firstgpu, i);
      cudaDeviceCanAccessPeer(&canAccessPeerx_0, i, firstgpu);
      if (verbose_flag) {
        printf(
            "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n",
            dprop0.name, firstgpu, dpropX.name, i,
            canAccessPeer0_x ? "Yes" : "No");
        printf(
            "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n",
            dpropX.name, i, dprop0.name, firstgpu,
            canAccessPeerx_0 ? "Yes" : "No");
      }
      if (canAccessPeer0_x == 0 || canAccessPeerx_0 == 0) {
        printf("Number of GPUs: %d\n", num_gpus);
        printf("Two or more SM 2.0 class GPUs are required for %s to run.\n",
               argv[0]);
        printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
        printf(
            "Peer to Peer access is not available between GPU%d <-> GPU%d, "
            "waiving test.\n",
            0, i);
        exit(EXIT_SUCCESS);
      } else {
        cudaSetDevice(firstgpu);
        if (verbose_flag) {
          printf("Granting access from %d to %d...\n", firstgpu, i);
        }
        cudaDeviceEnablePeerAccess(i, 0);
        cudaSetDevice(i);
        if (verbose_flag) {
          printf("Granting access from %d to %d...\n", i, firstgpu);
        }
        cudaDeviceEnablePeerAccess(firstgpu, 0);
        if (verbose_flag) {
          printf("Checking GPU %d and GPU %d for UVA capabilities...\n",
                 firstgpu, i);
        }
        const bool has_uva =
            (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
        if (verbose_flag) {
          printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, firstgpu,
                 (dprop0.unifiedAddressing ? "Yes" : "No"));
          printf("> %s (GPU%d) supports UVA: %s\n", dpropX.name, i,
                 (dpropX.unifiedAddressing ? "Yes" : "No"));
        }
        if (has_uva) {
          if (verbose_flag) {
            printf("Both GPUs can support UVA, enabling...\n");
          }
        } else {
          printf(
              "At least one of the two GPUs does NOT support UVA, waiving "
              "test.\n");
          exit(EXIT_SUCCESS);
        }
      }
    }
  }

  vars_gpu = (varsPerGPU*)malloc(num_gpus * sizeof(varsPerGPU));

  this->setDatasets(&datasets);
  this->setNDatasets(nMeasurementSets);
  g_datasets = &datasets;

  double deltax = RPDEG_D * DELTAX;  // radians
  double deltay = RPDEG_D * DELTAY;  // radians
  deltau = 1.0 / (M * deltax);
  deltav = 1.0 / (N * deltay);

  if (this->scheme == NULL) {
    this->scheme =
        Singleton<WeightingSchemeFactory>::Instance().CreateWeightingScheme(0);
  }

  if (this->gridding)
    this->scheme->setThreads(this->griddingThreads);

  this->scheme->configure(&robust_param);
  this->scheme->setModifyWeights(modify_weights);
  this->scheme->apply(datasets);

  if (this->gridding) {
    std::cout << "Doing gridding" << std::endl;
    this->ckernel->setSigmas(fabs(deltau), fabs(deltav));
    this->ckernel->buildKernel();
    this->ckernel->printCKernel();
    this->ckernel->initializeGCF(M, N, fabs(deltax), fabs(deltay));
    this->ckernel->printGCF();

    printf(
        "Using an antialiasing kernel %s of size (%d, %d) and support (%d, "
        "%d)\n",
        this->ckernel->getName().c_str(), this->ckernel->getm(),
        this->ckernel->getn(), this->ckernel->getSupportX(),
        this->ckernel->getSupportY());
    Gridder gridder(this->ckernel, this->getGriddingThreads());
    gridder.grid(datasets);
  } else {
    for (int d = 0; d < nMeasurementSets; d++)
      datasets[d].upload();
  }
}

void MFS::setDevice() {
  double deltax = RPDEG_D * DELTAX;  // radians
  double deltay = RPDEG_D * DELTAY;  // radians
  deltau = 1.0 / (M * deltax);
  deltav = 1.0 / (N * deltay);

  if (verbose_flag) {
    printf("MS files reading ");
    if (this->gridding) {
      printf("and gridding ");
    }
    printf("OK!\n");
    if (this->getVisNoise() < 0.0f) {
      printf("Beam noise wasn't provided by the user... Calculating...\n");
    }
  }

  // Estimates the noise in JY/BEAM, beam major, minor axis and angle in
  // degrees (each dataset contributes its beam/noise; we accumulate then
  // compute once).
  {
    double s_uu = 0.0, s_vv = 0.0, s_uv = 0.0;
    float sw = 0.0f;
    int tot_vis = 0;
    for (auto& dw : datasets) {
      dw.computeNoiseAndBeamContribution(&s_uu, &s_vv, &s_uv, &sw, &tot_vis);
    }
    total_visibilities = tot_vis;
    sum_weights = sw;
    gpuvmem::ms::beamNoiseFromSums(s_uu, s_vv, s_uv, sum_weights, &beam_bmaj,
                                   &beam_bmin, &beam_bpa, &this->vis_noise);
  }

  this->setTotalVisibilities(total_visibilities);

  for (int d = 0; d < nMeasurementSets; d++) {
    if (datasets[d].gpu.num_fields() == 0)
      datasets[d].upload();
    gpuvmem::ms::MeasurementSet& ms = datasets[d].ms;
    datasets[d].atten_image.resize(ms.num_fields());
    cudaSetDevice(firstgpu);
    for (size_t f = 0; f < ms.num_fields(); f++) {
      checkCudaErrors(cudaMalloc((void**)&datasets[d].atten_image[f],
                                 sizeof(float) * M * N));
      checkCudaErrors(cudaMemset(datasets[d].atten_image[f], 0,
                                 sizeof(float) * M * N));
    }
  }

  max_number_vis = 0;
  for (int d = 0; d < nMeasurementSets; d++) {
    size_t mc = datasets[d].gpu.max_chunk_count();
    if (mc > static_cast<size_t>(max_number_vis))
      max_number_vis = static_cast<int>(mc);
  }

  if (max_number_vis == 0) {
    printf("Max number of visibilities cannot be zero for image synthesis\n");
    exit(-1);
  }

  this->setMaxNumberVis(max_number_vis);

  printf("Estimated beam size: %e x %e (arcsec) / %lf (degrees)\n",
         beam_bmaj * 3600.0, beam_bmin * 3600.0, beam_bpa);
  printf("gpuvmem estimated beam size: %e x %e (arcsec) / %lf (degrees)\n",
         beam_bmaj * 1200.0, beam_bmin * 1200.0,
         beam_bpa);                      // ~ A third of the clean beam
  beam_bmaj = beam_bmaj / fabs(DELTAX);  // Beam major axis to pixels
  beam_bmin = beam_bmin / fabs(DELTAX);  // Beam minor axis to pixels
  noise_jypix =
      this->getVisNoise() / (PI_D * beam_bmaj * beam_bmin /
                             (4.0 * logf(2.0)));  // Estimating noise at FWHM

  /////////////////////////////////////////////////////CALCULATE DIRECTION
  /// COSINES/////////////////////////////////////////////////
  std::cout << "Checking frames..." << std::endl;
  double raimage = ra * RPDEG_D;
  double decimage = dec * RPDEG_D;

  if (verbose_flag) {
    printf("Original right ascension and declination\n");
    printf("FITS: Ra: (%.16e, %.16e) rad\n", raimage, decimage);
    printf("FITS: Center pix: (%lf,%lf)\n", crpix1 - 1, crpix2 - 1);
  }

  double lobs, mobs, lphs, mphs;
  double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs,
      dcosines_m_pix_phs;
  for (int d = 0; d < nMeasurementSets; d++) {
    if (verbose_flag)
      printf("Dataset: %s\n", datasets[d].name);
    gpuvmem::ms::MeasurementSet& ms = datasets[d].ms;
    for (size_t f = 0; f < ms.num_fields(); f++) {
      gpuvmem::ms::Field& field = ms.field(f);
      gpuvmem::ms::FieldMetadata& fmeta = field.metadata();
      direccos(field.reference_dir()[0], field.reference_dir()[1],
               raimage, decimage, &lobs, &mobs);
      direccos(field.phase_dir()[0], field.phase_dir()[1],
               raimage, decimage, &lphs, &mphs);

      dcosines_l_pix_phs = lphs / deltax;  // Radians to pixels
      dcosines_m_pix_phs = mphs / deltay;  // Radians to pixels

      if (verbose_flag) {
        printf("Ref: l (pix): %e, m (pix): %e\n", dcosines_l_pix_ref,
               dcosines_m_pix_ref);
        printf("Phase: l (pix): %e, m (pix): %e\n", dcosines_l_pix_phs,
               dcosines_m_pix_phs);
      }
      fmeta.ref_xobs_pix = static_cast<float>(dcosines_l_pix_phs + (crpix1 - 1.0));
      fmeta.ref_yobs_pix = static_cast<float>(dcosines_m_pix_phs + (crpix2 - 1.0));
      fmeta.phs_xobs_pix = static_cast<float>(dcosines_l_pix_phs + (crpix1 - 1.0));
      fmeta.phs_yobs_pix = static_cast<float>(dcosines_m_pix_phs + (crpix2 - 1.0));

      if (verbose_flag) {
        printf(
            "Ref: Field %zu - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), "
            "y0: %f (pix)\n",
            f, field.reference_dir()[0], field.reference_dir()[1],
            fmeta.ref_xobs_pix, fmeta.ref_yobs_pix);
        printf(
            "Phase: Field %zu - Ra: %.16e (rad), dec: %.16e (rad), x0: %f "
            "(pix), y0: %f (pix)\n",
            f, field.phase_dir()[0], field.phase_dir()[1],
            fmeta.phs_xobs_pix, fmeta.phs_yobs_pix);
      }

      if (fmeta.ref_xobs_pix < 0 || fmeta.ref_xobs_pix >= M ||
          fmeta.ref_yobs_pix < 0 || fmeta.ref_yobs_pix >= N) {
        printf("Dataset: %s\n", datasets[d].name.c_str());
        printf(
            "Pointing reference center (%f,%f) is outside the range of the "
            "image\n",
            fmeta.ref_xobs_pix, fmeta.ref_yobs_pix);
        goToError();
      }

      if (fmeta.phs_xobs_pix < 0 || fmeta.phs_xobs_pix >= M ||
          fmeta.phs_yobs_pix < 0 || fmeta.phs_yobs_pix >= N) {
        printf("Dataset: %s\n", datasets[d].name.c_str());
        printf(
            "Pointing phase center (%f,%f) is outside the range of the image\n",
            fmeta.phs_xobs_pix, fmeta.phs_yobs_pix);
        goToError();
      }
    }
  }
  ////////////////////////////////////////////////////////MAKE STARTING
  /// IMAGE////////////////////////////////////////////////////////

  host_I = (float*)malloc(M * N * sizeof(float) * image_count);

  for (int k = 0; k < image_count; k++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        host_I[N * M * k + N * i + j] = initial_values[k];
      }
    }
  }

  ////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR
  /// DEVICE///////////////////////////////////////////////////
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    checkCudaErrors(
        cudaMalloc(&vars_gpu[g].device_V, sizeof(cufftComplex) * M * N));
    checkCudaErrors(
        cudaMalloc(&vars_gpu[g].device_I_nu, sizeof(cufftComplex) * M * N));
    checkCudaErrors(
        cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float) * max_number_vis));
    checkCudaErrors(
        cudaMalloc(&vars_gpu[g].device_dchi2, sizeof(float) * M * N));

    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex) * M * N));
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_I_nu, 0, sizeof(cufftComplex) * M * N));
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float) * max_number_vis));
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_dchi2, 0, sizeof(float) * M * N));
  }

  cudaSetDevice(firstgpu);

  checkCudaErrors(
      cudaMalloc((void**)&device_Image, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMemset(device_Image, 0, sizeof(float) * M * N * image_count));

  checkCudaErrors(cudaMemcpy(device_Image, host_I,
                             sizeof(float) * N * M * image_count,
                             cudaMemcpyHostToDevice));

  checkCudaErrors(
      cudaMalloc((void**)&device_noise_image, sizeof(float) * M * N));
  checkCudaErrors(cudaMemset(device_noise_image, 0, sizeof(float) * M * N));

  checkCudaErrors(
      cudaMalloc((void**)&device_weight_image, sizeof(float) * M * N));
  checkCudaErrors(cudaMemset(device_weight_image, 0, sizeof(float) * M * N));

  if (radius_mask)
    checkCudaErrors(
        cudaMalloc((void**)&device_distance_image, sizeof(float) * M * N));

  /////////// MAKING IMAGE OBJECT /////////////
  image = new Image(device_Image, image_count, M, N);
  image->set_pixel_scale_ra_deg(DELTAX);
  image->set_pixel_scale_dec_deg(DELTAY);
  // Set minimal pixel values from initial_values (before eta multiplication)
  // minimal_pixel_values was declared in configure() and stored as member variable
  image->setMinimalPixelValues(this->minimal_pixel_values);
  // Set global I pointer for backward compatibility (used as fallback in line searchers)
  // Note: Line searchers prefer using this->image from optimizer, but fall back to extern I if needed
  I = image;
  // Configure Chi2's ImageProcessor from Image geometry (so ip uses image->getM(), getN(), getImageCount())
  if (this->getOptimizator() && this->getOptimizator()->getObjectiveFunction()) {
    Fi* chi2_fi = this->getOptimizator()->getObjectiveFunction()->getFiByName("Chi2");
    Chi2* chi2_term = (chi2_fi ? dynamic_cast<Chi2*>(chi2_fi) : nullptr);
    if (chi2_term)
      chi2_term->configureImage(image);
  }
  imageMap* functionPtr = (imageMap*)malloc(sizeof(imageMap) * image_count);
  image->setFunctionMapping(functionPtr);

  for (int i = 0; i < image_count; i++) {
    if (nopositivity) {
      functionPtr[i].evaluateXt = defaultEvaluateXt;
      functionPtr[i].newP = defaultNewP;
    } else {
      if (!i) {
        functionPtr[i].evaluateXt = particularEvaluateXt;
        functionPtr[i].newP = particularNewP;
      } else {
        functionPtr[i].evaluateXt = defaultEvaluateXt;
        functionPtr[i].newP = defaultNewP;
      }
    }
  }

  initFFT(vars_gpu, M, N, firstgpu, num_gpus);

  // Time is taken from first kernels
  t = clock();
  start = omp_get_wtime();
  for (int d = 0; d < nMeasurementSets; d++) {
    const gpuvmem::ms::MeasurementSetMetadata& meta = datasets[d].ms.metadata();
    if (meta.num_antennas() == 0) continue;
    const gpuvmem::ms::Antenna& ant0 = meta.antenna(0);
    int primary_beam_int =
        (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0;
    cudaSetDevice(firstgpu);
    for (size_t f = 0; f < datasets[d].ms.num_fields(); f++) {
      const gpuvmem::ms::FieldMetadata& fmeta =
          datasets[d].ms.field(f).metadata();
      total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(
          datasets[d].atten_image[f],
          ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
          nu_0, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix, DELTAX, DELTAY, N,
          primary_beam_int);
      checkCudaErrors(cudaDeviceSynchronize());

      if (print_images) {
        std::string atten_name = "dataset_" + std::to_string(d) + "_atten";
        ioImageHandler->printNotNormalizedImageIteration(
            datasets[d].atten_image[f], atten_name.c_str(), "",
            static_cast<int>(f), 0, true);
      }
    }
  }

  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (size_t f = 0; f < datasets[d].ms.num_fields(); f++) {
      weight_image<<<numBlocksNN, threadsPerBlockNN>>>(
          device_weight_image, datasets[d].atten_image[f], N);
      checkCudaErrors(cudaDeviceSynchronize());

      if (radius_mask) {
        const gpuvmem::ms::FieldMetadata& fmeta =
            datasets[d].ms.field(f).metadata();
        distance_image<<<numBlocksNN, threadsPerBlockNN>>>(
            device_distance_image, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix,
            4.5e-05, DELTAX, DELTAY, N);
        checkCudaErrors(cudaDeviceSynchronize());
      }
    }
  }

  float* host_weight_image = (float*)malloc(M * N * sizeof(float));
  checkCudaErrors(cudaMemcpy2D(host_weight_image, sizeof(float),
                               device_weight_image, sizeof(float),
                               sizeof(float), M * N, cudaMemcpyDeviceToHost));
  float max_weight =
      *std::max_element(host_weight_image, host_weight_image + (M * N));

  noise_image<<<numBlocksNN, threadsPerBlockNN>>>(
      device_noise_image, device_weight_image, max_weight, noise_jypix, N);
  checkCudaErrors(cudaDeviceSynchronize());
  if (print_images) {
    ioImageHandler->printNotNormalizedImage(device_noise_image, "noise.fits",
                                            "", 0, 0, true);
    if (radius_mask)
      ioImageHandler->printNotNormalizedImage(device_distance_image,
                                              "distance.fits", "", 0, 0, true);
  }

  float* host_noise_image = (float*)malloc(M * N * sizeof(float));
  checkCudaErrors(cudaMemcpy2D(host_noise_image, sizeof(float),
                               device_noise_image, sizeof(float), sizeof(float),
                               M * N, cudaMemcpyDeviceToHost));
  float noise_min =
      *std::min_element(host_noise_image, host_noise_image + (M * N));

  this->fg_scale = noise_min;
  noise_cut = noise_cut * noise_min;
  // MINPIX is set from the -z option (first initial value) at line 181
  // Note: 0.0 is a valid value for MINPIX (user may want values >= 0)
  // So we don't check if MINPIX == 0.0f, as that would incorrectly overwrite
  // a valid user-specified value of 0.0
  if (verbose_flag) {
    printf("fg_scale = %e\n", this->fg_scale);
    printf("noise (Jy/pix) = %e\n", noise_jypix);
    printf("MINPIX = %e (from -z option)\n", MINPIX);
  }

  std::vector<float> u_mask;
  if (variables.user_mask != "NULL") {
    u_mask = ioImageHandler->read_data_float_FITS(variables.user_mask);
    checkCudaErrors(cudaMemcpy2D(device_noise_image, sizeof(float),
                                 u_mask.data(), sizeof(float), sizeof(float),
                                 M * N, cudaMemcpyHostToDevice));
  } else if (radius_mask)
    checkCudaErrors(cudaMemcpy2D(
        device_noise_image, sizeof(float), device_distance_image, sizeof(float),
        sizeof(float), M * N, cudaMemcpyDeviceToDevice));

  free(host_noise_image);
  free(host_weight_image);
  cudaFree(device_weight_image);
  if (radius_mask)
    cudaFree(device_distance_image);
  for (int d = 0; d < nMeasurementSets; d++) {
    for (size_t f = 0; f < datasets[d].atten_image.size(); f++) {
      cudaFree(datasets[d].atten_image[f]);
    }
    datasets[d].atten_image.clear();
  }
};

void MFS::clearRun() {
  for (int d = 0; d < nMeasurementSets; d++) {
    datasets[d].gpu.zero_model_and_residual();
  }

  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex) * M * N));
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_I_nu, 0, sizeof(cufftComplex) * M * N));
  }

  cudaSetDevice(firstgpu);
  checkCudaErrors(cudaMemcpy(device_Image, host_I,
                             sizeof(float) * N * M * image_count,
                             cudaMemcpyHostToDevice));
};

void MFS::run() {
  optimizer->getObjectiveFunction()->setIo(ioImageHandler);

  Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");

  if (NULL != chi2 && chi2->getNormalize())
    this->fg_scale = 1.0f;

  if (NULL != chi2)
    chi2->setFgScale(this->fg_scale);

  if (this->gridding) {
    if (NULL != chi2)
      chi2->setCKernel(this->ckernel);
  }

  // Pre-compute effective number of samples before optimization starts
  // This avoids recalculating N_eff on every iteration
  bool normalize = (NULL != chi2 && chi2->getNormalize());
  if (normalize) {
    printf("Pre-computing effective number of samples...\n");
    precomputeNeff(true);
  }

  printf("\n\nStarting optimizer\n");
  // Four modes: joint (I_nu_0+alpha), block (alternate), one (single image), alpha_static (I_nu_0 only, alpha fixed)
  const std::string& mode = variables.optimization_mode;

  if (image_count == 1) {
    // One image mode: single run (flag 0 = single block)
    if (verbose_flag)
      printf("Optimization mode: one (single image)\n");
    optimizer->setImage(image);
    optimizer->setFlag(0);
    optimizer->optimize();
  } else if (image_count == 2) {
    optimizer->setImage(image);
    if (mode == "joint") {
      if (verbose_flag)
        printf("Optimization mode: joint (I_nu_0 + alpha together)\n");
      optimizer->setFlag(-1);  // -1 = joint: gradient for all images
      optimizer->optimize();
    } else if (mode == "alpha_static") {
      if (verbose_flag)
        printf("Optimization mode: alpha_static (I_nu_0 only, alpha fixed)\n");
      optimizer->setFlag(0);  // I_nu_0 only
      optimizer->optimize();
    } else if (mode == "block" && this->Order != NULL) {
      if (verbose_flag)
        printf("Optimization mode: block (alternating I_nu_0, alpha, ...)\n");
      (this->Order)(optimizer, image);
    } else {
      // block with Order==NULL, or unknown mode: fallback to alternating
      if (verbose_flag)
        printf("Optimization mode: block (alternating I_nu_0, alpha, ...)\n");
      optimizer->setFlag(0);
      optimizer->optimize();
      optimizer->setFlag(1);
      optimizer->optimize();
      optimizer->setFlag(2);
      optimizer->optimize();
      optimizer->setFlag(3);
      optimizer->optimize();
    }
  } else if (this->Order != NULL) {
    (this->Order)(optimizer, image);
  } else if (imagesChanged) {
    optimizer->setImage(image);
    optimizer->optimize();
  }

  float chi2_final = 0.0f;
  float final_S = 0.0f;
  float lambda_S = 0.0f;
  bool normalize_enabled = false;

  if (NULL != chi2) {
    chi2_final = chi2->get_fivalue();
    normalize_enabled = chi2->getNormalize();
  }
  Fi* entropy = optimizer->getObjectiveFunction()->getFiByName("Entropy");
  if (NULL != entropy) {
    final_S = entropy->get_fivalue();
    lambda_S = entropy->getPenalizationFactor();
  }

  t = clock() - t;
  end = omp_get_wtime();
  printf("Minimization ended successfully\n\n");
  printf("Iterations: %d\n", optimizer->getCurrentIteration());
  printf("chi2: %f\n", 2.0f * chi2_final);
  printf("0.5*chi2: %f\n", chi2_final);
  printf("Total visibilities: %d\n", total_visibilities);

  if (normalize_enabled) {
    // chi2_final is already normalized by N_eff (effective number of samples)
    // This IS the reduced chi-squared using effective number of samples
    printf("Reduced-chi2 (N_eff normalized): %f\n", chi2_final);
  } else {
    // chi2_final is not normalized, so we divide by total_visibilities or
    // sum_weights
    printf("Reduced-chi2 (Num visibilities): %f\n",
           chi2_final / total_visibilities);
    printf("Reduced-chi2 (Weights sum): %f\n", chi2_final / sum_weights);
  }
  printf("S: %f\n", final_S);
  if (reg_term != 1) {
    printf("Normalized S: %f\n", final_S / (M * N));
  } else {
    printf("Normalized S: %f\n", final_S / (M * M * N * N));
  }
  printf("lambda*S: %f\n\n", lambda_S * final_S);
  double time_taken = ((double)t) / CLOCKS_PER_SEC;
  double wall_time = end - start;
  printf("Total CPU time: %lf\n", time_taken);
  printf("Wall time: %lf\n\n\n", wall_time);

  if (variables.ofile != "NULL") {
    FILE* outfile = fopen(variables.ofile.c_str(), "w");
    if (outfile == NULL) {
      printf("Error opening output file!\n");
      goToError();
    }

    fprintf(outfile, "Iterations: %d\n", optimizer->getCurrentIteration());
    fprintf(outfile, "chi2: %f\n", 2.0f * chi2_final);
    fprintf(outfile, "0.5*chi2: %f\n", chi2_final);
    fprintf(outfile, "Total visibilities: %d\n", total_visibilities);

    if (normalize_enabled) {
      // chi2_final is already normalized by N_eff (effective number of samples)
      // This IS the reduced chi-squared using effective number of samples
      fprintf(outfile, "Reduced-chi2 (N_eff normalized): %f\n", chi2_final);
    } else {
      // chi2_final is not normalized, so we divide by total_visibilities or
      // sum_weights
      fprintf(outfile, "Reduced-chi2 (Num visibilities): %f\n",
              chi2_final / total_visibilities);
      fprintf(outfile, "Reduced-chi2 (Weights sum): %f\n",
              chi2_final / sum_weights);
    }
    fprintf(outfile, "S: %f\n", final_S);
    if (reg_term != 1) {
      fprintf(outfile, "Normalized S: %f\n", final_S / (M * N));
    } else {
      fprintf(outfile, "Normalized S: %f\n", final_S / (M * M * N * N));
    }
    fprintf(outfile, "lambda*S: %f\n", lambda_S * final_S);
    fprintf(outfile, "Wall time: %lf", wall_time);
    fclose(outfile);
  }
};

void MFS::writeImages() {
  printf("Saving final image to disk\n");
  
  if (IoOrderEnd == NULL) {
    ioImageHandler->printNotPathImage(image->getImage(), "JY/PIXEL",
                                      optimizer->getCurrentIteration(), 0,
                                      this->fg_scale, true);
    if (print_images)
      ioImageHandler->printNotNormalizedImage(
          image->getImage(), "alpha.fits", "", optimizer->getCurrentIteration(),
          1, true);
  } else {
    (IoOrderEnd)(image->getImage(), ioImageHandler);
  }

  if (print_errors) /* flag for print error image */
  {
    if (this->error == NULL) {
      this->error = createObject<Error, std::string>("SecondDerivateError");
      // Set optimizer reference so SecondDerivateError can access Chi2 for
      // fg_scale
      SecondDerivateError* sde =
          dynamic_cast<SecondDerivateError*>(this->error);
      if (sde != NULL) {
        sde->setOptimizer(this->optimizer);
      }
    }
    /* code to calculate error */
    /* make void * params */
    printf("Calculating Error Images\n");
    this->error->calculateErrorImage(this->image, *this->getDatasets());
    if (IoOrderError == NULL) {
      if (print_images) {
        int image_count = image->getImageCount();
        if (image_count == 2) {
          // MFS: 0 = σ(I_nu_0), 1 = σ(alpha), 2 = Cov, 3 = ρ.
          ioImageHandler->printNormalizedImage(
              image->getErrorImage(), "error_Inu_0.fits", "JY/PIXEL",
              optimizer->getCurrentIteration(), 0, this->fg_scale, true);
          ioImageHandler->printNotNormalizedImage(
              image->getErrorImage(), "error_alpha_0.fits", "",
              optimizer->getCurrentIteration(), 1, true);
          ioImageHandler->printNormalizedImage(
              image->getErrorImage(), "error_cov_Inu_0_alpha.fits", "JY/PIXEL",
              optimizer->getCurrentIteration(), 2, this->fg_scale, true);
          ioImageHandler->printNotNormalizedImage(
              image->getErrorImage(), "error_rho_Inu_0_alpha.fits", "",
              optimizer->getCurrentIteration(), 3, true);
        } else {
          // Stokes/single: one σ per image plane (Jy/pixel with fg_scale).
          for (int p = 0; p < image_count; p++) {
            char fname[64];
            snprintf(fname, sizeof(fname), "error_stokes_%d.fits", p);
            ioImageHandler->printNormalizedImage(
                image->getErrorImage(), fname, "JY/PIXEL",
                optimizer->getCurrentIteration(), p, this->fg_scale, true);
          }
        }
      }

    } else {
      (IoOrderError)(image->getErrorImage(), ioImageHandler);
    }
  }
};

void MFS::writeResiduals() {
  // Restoring the weights to the original
  printf("Transferring residuals to host memory\n");
  if (!this->gridding) {
    this->scheme->restoreWeights(*this->getDatasets());
  } else {
    double deltax = RPDEG_D * DELTAX;  // radians
    double deltay = RPDEG_D * DELTAY;  // radians
    deltau = 1.0 / (M * deltax);
    deltav = 1.0 / (N * deltay);

    printf(
        "Visibilities are gridded, we will need to de-grid to save them in a "
        "Measurement Set File\n");
    // Native degridding: compute model visibilities from final image on
    // MeasurementSet + ChunkedVisibilityGPU; download to ms; refresh legacy view.
    ImageProcessor* ip = new ImageProcessor();
    ip->configure(image);
    if (this->ckernel != NULL) {
      ip->setCKernel(this->ckernel);
    }
    Gridder gridder(this->ckernel, this->getGriddingThreads());
    for (int d = 0; d < nMeasurementSets; d++) {
      // Upload original (ungridded) ms to gpu so degridding fills Vm at
      // original positions; gpu currently holds gridded data from do_gridding.
      datasets[d].upload();
      gridder.degrid(datasets[d], image->getImage(), ip);
      datasets[d].download();
    }
    delete ip;

    int max_orig = 0;
    for (int d = 0; d < nMeasurementSets; d++) {
      size_t mc = datasets[d].gpu.max_chunk_count();
      if (mc > static_cast<size_t>(max_orig)) max_orig = static_cast<int>(mc);
    }
    if (max_orig > max_number_vis) {
      max_number_vis = max_orig;
      for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(firstgpu + g);
        cudaFree(vars_gpu[g].device_chi2);
        checkCudaErrors(cudaMalloc(&vars_gpu[g].device_chi2,
                                   sizeof(float) * max_number_vis));
      }
    }

    Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
    float res = chi2->calcFi(image->getImage());
    printf("Non-gridded chi2 after de-gridding using convolution kernel %f\n",
           res);
  }

  for (int d = 0; d < nMeasurementSets; d++) {
    datasets[d].download();
  }
  printf("Saving residuals and model to MS...\n");
  std::unique_ptr<gpuvmem::ms::MSWriter> writer =
      gpuvmem::ms::create_ms_writer();
  gpuvmem::ms::MSWriteOptions opts;
  opts.write_columns = {gpuvmem::ms::DataColumn::MODEL_DATA};
  opts.write_residual = true;
  opts.residual_column_name = "RESIDUAL_DATA";
  opts.update_weights = true;
  for (int d = 0; d < nMeasurementSets; d++) {
    ioVisibilitiesHandler->copy(datasets[d].name.c_str(),
                                datasets[d].oname.c_str());
    gpuvmem::ms::MeasurementSet& ms = datasets[d].ms;
    if (ms.storage_mode() == gpuvmem::ms::StorageMode::Stokes) {
      gpuvmem::ms::stokes_to_correlations(ms);
    }
    if (!writer->write(datasets[d].oname, ms, opts)) {
      std::fprintf(stderr, "MFS: MSWriter failed for %s\n",
                   datasets[d].oname.c_str());
    }
  }
  printf("Residuals and model visibilities saved.\n");
};

void MFS::unSetDevice() {
  printf("Freeing device memory\n");
  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    datasets[d].clear();
    for (size_t f = 0; f < datasets[d].atten_image.size(); f++) {
      cudaFree(datasets[d].atten_image[f]);
    }
    datasets[d].atten_image.clear();
  }

  printf("Freeing cuFFT plans\n");
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    cufftDestroy(vars_gpu[g].plan);
  }

  printf("Freeing host memory\n");
  cudaSetDevice(firstgpu);
  cudaFree(device_Image);

  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    cudaFree(vars_gpu[g].device_V);
    cudaFree(vars_gpu[g].device_I_nu);
  }

  cudaSetDevice(firstgpu);

  cudaFree(device_noise_image);

  cudaFree(device_dphi);
  cudaFree(device_dchi2_total);
  cudaFree(device_dS);

  cudaFree(device_S);

  // Disabling UVA
  if (num_gpus > 1) {
    for (int i = firstgpu + 1; i < num_gpus + firstgpu; i++) {
      cudaSetDevice(firstgpu);
      cudaDeviceDisablePeerAccess(i);
      cudaSetDevice(i);
      cudaDeviceDisablePeerAccess(firstgpu);
    }

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice((i % num_gpus) + firstgpu);
      cudaDeviceReset();
    }
  }
  free(host_I);

  for (int i = 0; i < nMeasurementSets; i++) {
    datasets[i].name.clear();
    datasets[i].oname.clear();
  }
};

namespace {
Synthesizer* CreateMFS() {
  return new MFS;
}
const std::string name = "MFS";
const bool RegisteredMFS =
    registerCreationFunction<Synthesizer, std::string>(name, CreateMFS);
};  // namespace
