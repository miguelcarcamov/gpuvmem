#include "imageProcessor.cuh"
#include "mfs.cuh"

long M, N, numVisibilities;

float *device_Image, *device_dphi, *device_dchi2_total, *device_dS, *device_S,
    *device_noise_image, *device_weight_image, *device_distance_image;
float noise_cut, MINPIX, minpix, random_probability = 1.0;
float noise_jypix, fg_scale, eta, robust_param;
float *host_I, sum_weights, *penalizators;
double beam_bmaj, beam_bmin, beam_bpa;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int status_mod_in;
int multigpu, firstgpu, reg_term, total_visibilities, image_count,
    nPenalizators, nMeasurementSets = 0, max_number_vis;

std::string msinput, msoutput, modinput, mempath, out_image, output;
float nu_0, threshold;
extern int num_gpus;

double ra, dec, crpix1, crpix2, DELTAX, DELTAY, deltau, deltav;

std::vector<float> initial_values;
std::vector<MSDataset> datasets;

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
  ioVisibilitiesHandler->setGridding(variables.gridding);
  this->setGriddingThreads(variables.gridding);
  nu_0 = variables.nu_0;
  robust_param = variables.robust_param;
  threshold = variables.threshold * 5.0;
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
    datasets.push_back(MSDataset());
    datasets[i].name =
        (char*)malloc((string_values[i].length() + 1) * sizeof(char));
    datasets[i].oname =
        (char*)malloc((s_output_values[i].length() + 1) * sizeof(char));
    strcpy(datasets[i].name, string_values[i].c_str());
    strcpy(datasets[i].oname, s_output_values[i].c_str());
  }

  string_values.clear();
  s_output_values.clear();

  if (variables.initial_values != "NULL") {
    string_values = countAndSeparateStrings(variables.initial_values, ",");
    image_count = string_values.size();
  } else {
    printf("Initial values for image/s were not provided\n");
    print_help();
    exit(-1);
  }

  for (int i = 0; i < image_count; i++) {
    if (i == 0) {
      initial_values.push_back(std::stof(string_values[i]) * -1.0f * eta);
    } else {
      initial_values.push_back(std::stof(string_values[i]));
    }
  }

  string_values.clear();
  if (image_count == 1) {
    initial_values.push_back(0.0f);
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

  if (verbose_flag) {
    printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("Number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&dprop[i], i));
      printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i,
             dprop[i].name, (IsGPUCapableP2P(&dprop[i]) ? "IS " : "NOT"));
      printf("> Memory Clock Rate (KHz): %d\n", dprop[i].memoryClockRate);
      printf("> Memory Bus Width (bits): %d\n", dprop[i].memoryBusWidth);
      printf("> Peak Memory Bandwidth (GB/s): %f\n",
             2.0 * dprop[i].memoryClockRate * (dprop[i].memoryBusWidth / 8) /
                 1.0e6);
      printf("> Total Global Memory (GB): %f\n",
             dprop[i].totalGlobalMem / pow(2, 30));
      printf("-----------------------------------------------------------\n");
    }
    printf("-----------------------------------------------------------\n\n");
  }

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
  std::vector<float> ms_max_blength;
  std::vector<float> ms_min_blength;
  std::vector<float> ms_uvmax_wavelength;
  for (int d = 0; d < nMeasurementSets; d++) {
    ioVisibilitiesHandler->read(datasets[d].name, datasets[d].antennas,
                                datasets[d].fields, &datasets[d].data);
    ms_ref_freqs.push_back(datasets[d].data.ref_freq);
    ms_max_freqs.push_back(datasets[d].data.max_freq);
    ms_max_blength.push_back(datasets[d].data.max_blength);
    ms_min_blength.push_back(datasets[d].data.min_blength);
    ms_uvmax_wavelength.push_back(datasets[d].data.uvmax_wavelength);
    printf("Dataset %d: %s - Antenna diameter: %.3f metres\n", d,
           datasets[d].name, datasets[d].antennas[0].antenna_diameter);
  }

  /*
     Calculating theoretical resolution
   */
  float max_freq = *max_element(ms_max_freqs.begin(), ms_max_freqs.end());
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
        "Reference frequency not provided. It will be calculated as the median "
        "of all the arrays of frequencies.\n");
    nu_0 = median(ms_ref_freqs);
  }
  printf("Reference frequency: %e Hz\n", nu_0);
  double deltau_theo = 2.0 * max_uvmax_wavelength / (M - 1);
  double deltax_theo = 1.0 / (M * deltau_theo) / RPARCSEC;
  printf("The pixel size has to be less or equal to %lf arcsec\n", deltax_theo);
  printf("Actual pixel size is %lf arcsec\n", DELTAX * 3600.0);

  if (verbose_flag) {
    for (int i = 0; i < nMeasurementSets; i++) {
      printf("Dataset %d: %s\n", i, datasets[i].name);
      printf("\tNumber of fields = %d\n", datasets[i].data.nfields);
      printf("\tNumber of frequencies = %d\n",
             datasets[i].data.total_frequencies);
      printf("\tNumber of correlations = %d\n", datasets[i].data.nstokes);
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
        if (datasets[d].data.total_frequencies > max_nfreq)
          max_nfreq = datasets[d].data.total_frequencies;
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

  this->visibilities = new Visibilities();
  this->visibilities->setMSDataset(datasets);
  this->visibilities->setNDatasets(nMeasurementSets);

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
    for (int d = 0; d < nMeasurementSets; d++)
      do_gridding(datasets[d].fields, &datasets[d].data, deltau, deltav, M, N,
                  this->ckernel, this->getGriddingThreads());
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
  // degrees.
  sum_weights = calculateNoiseAndBeam(datasets, &total_visibilities,
                                      variables.blockSizeV, &beam_bmaj,
                                      &beam_bmin, &beam_bpa, &this->vis_noise);

  this->visibilities->setTotalVisibilities(total_visibilities);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      cudaSetDevice(firstgpu);
      checkCudaErrors(cudaMalloc((void**)&datasets[d].fields[f].atten_image,
                                 sizeof(float) * M * N));
      checkCudaErrors(cudaMemset(datasets[d].fields[f].atten_image, 0,
                                 sizeof(float) * M * N));
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        cudaSetDevice((i % num_gpus) + firstgpu);
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          checkCudaErrors(cudaMalloc(
              &datasets[d].fields[f].device_visibilities[i][s].uvw,
              sizeof(double3) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMalloc(
              &datasets[d].fields[f].device_visibilities[i][s].Vo,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMalloc(
              &datasets[d].fields[f].device_visibilities[i][s].weight,
              sizeof(float) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMalloc(
              &datasets[d].fields[f].device_visibilities[i][s].Vm,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMalloc(
              &datasets[d].fields[f].device_visibilities[i][s].Vr,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMemcpy(
              datasets[d].fields[f].device_visibilities[i][s].uvw,
              datasets[d].fields[f].visibilities[i][s].uvw.data(),
              sizeof(double3) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
              cudaMemcpyHostToDevice));

          checkCudaErrors(cudaMemcpy(
              datasets[d].fields[f].device_visibilities[i][s].weight,
              datasets[d].fields[f].visibilities[i][s].weight.data(),
              sizeof(float) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
              cudaMemcpyHostToDevice));

          checkCudaErrors(cudaMemcpy(
              datasets[d].fields[f].device_visibilities[i][s].Vo,
              datasets[d].fields[f].visibilities[i][s].Vo.data(),
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
              cudaMemcpyHostToDevice));

          checkCudaErrors(cudaMemset(
              datasets[d].fields[f].device_visibilities[i][s].Vr, 0,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMemset(
              datasets[d].fields[f].device_visibilities[i][s].Vm, 0,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        }
      }
    }
  }

  max_number_vis = 0;
  for (int d = 0; d < nMeasurementSets; d++) {
    if (datasets[d].data.max_number_visibilities_in_channel_and_stokes >
        max_number_vis)
      max_number_vis =
          datasets[d].data.max_number_visibilities_in_channel_and_stokes;
  }

  if (max_number_vis == 0) {
    printf("Max number of visibilities cannot be zero for image synthesis\n");
    exit(-1);
  }

  this->visibilities->setMaxNumberVis(max_number_vis);

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
  double raimage = ra * RPDEG_D;
  double decimage = dec * RPDEG_D;

  if (verbose_flag) {
    printf("FITS: Ra: %.16e (rad), dec: %.16e (rad)\n", raimage, decimage);
    printf("FITS: Center pix: (%lf,%lf)\n", crpix1 - 1, crpix2 - 1);
  }

  double lobs, mobs, lphs, mphs;
  double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs,
      dcosines_m_pix_phs;
  for (int d = 0; d < nMeasurementSets; d++) {
    if (verbose_flag)
      printf("Dataset: %s\n", datasets[d].name);
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      direccos(datasets[d].fields[f].ref_ra, datasets[d].fields[f].ref_dec,
               raimage, decimage, &lobs, &mobs);
      direccos(datasets[d].fields[f].phs_ra, datasets[d].fields[f].phs_dec,
               raimage, decimage, &lphs, &mphs);

      dcosines_l_pix_ref = lobs / -deltax;       // Radians to pixels
      dcosines_m_pix_ref = mobs / fabs(deltay);  // Radians to pixels

      dcosines_l_pix_phs = lphs / -deltax;       // Radians to pixels
      dcosines_m_pix_phs = mphs / fabs(deltay);  // Radians to pixels

      if (verbose_flag) {
        printf("Ref: l (pix): %e, m (pix): %e\n", dcosines_l_pix_ref,
               dcosines_m_pix_ref);
        printf("Phase: l (pix): %e, m (pix): %e\n", dcosines_l_pix_phs,
               dcosines_m_pix_phs);
      }

      datasets[d].fields[f].ref_xobs = (crpix1 - 1.0f) + dcosines_l_pix_ref;
      datasets[d].fields[f].ref_yobs = (crpix2 - 1.0f) + dcosines_m_pix_ref;

      datasets[d].fields[f].phs_xobs = (crpix1 - 1.0f) + dcosines_l_pix_phs;
      datasets[d].fields[f].phs_yobs = (crpix2 - 1.0f) + dcosines_m_pix_phs;

      if (verbose_flag) {
        printf(
            "Ref: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), "
            "y0: %f (pix)\n",
            f, datasets[d].fields[f].ref_ra, datasets[d].fields[f].ref_dec,
            datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs);
        printf(
            "Phase: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f "
            "(pix), y0: %f (pix)\n",
            f, datasets[d].fields[f].phs_ra, datasets[d].fields[f].phs_dec,
            datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
      }

      if (datasets[d].fields[f].ref_xobs < 0 ||
          datasets[d].fields[f].ref_xobs >= M ||
          datasets[d].fields[f].ref_xobs < 0 ||
          datasets[d].fields[f].ref_yobs >= N) {
        printf("Dataset: %s\n", datasets[d].name);
        printf(
            "Pointing reference center (%f,%f) is outside the range of the "
            "image\n",
            datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs);
        goToError();
      }

      if (datasets[d].fields[f].phs_xobs < 0 ||
          datasets[d].fields[f].phs_xobs >= M ||
          datasets[d].fields[f].phs_xobs < 0 ||
          datasets[d].fields[f].phs_yobs >= N) {
        printf("Dataset: %s\n", datasets[d].name);
        printf(
            "Pointing phase center (%f,%f) is outside the range of the image\n",
            datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
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
  image = new Image(device_Image, image_count);
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
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
            hermitianSymmetry<<<
                datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                datasets[d]
                    .fields[f]
                    .device_visibilities[i][s]
                    .threadsPerBlockUV>>>(
                datasets[d].fields[f].device_visibilities[i][s].uvw,
                datasets[d].fields[f].device_visibilities[i][s].Vo,
                datasets[d].fields[f].nu[i],
                datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
            checkCudaErrors(cudaDeviceSynchronize());
          }
        }
      }
    }

    cudaSetDevice(firstgpu);

    for (int f = 0; f < datasets[d].data.nfields; f++) {
      total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(
          datasets[d].fields[f].atten_image,
          datasets[d].antennas[0].antenna_diameter,
          datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff,
          nu_0, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs,
          DELTAX, DELTAY, N, datasets[d].antennas[0].primary_beam);
      checkCudaErrors(cudaDeviceSynchronize());

      if (print_images) {
        std::string atten_name = "dataset_" + std::to_string(d) + "_atten";
        ioImageHandler->printNotNormalizedImageIteration(
            datasets[d].fields[f].atten_image, atten_name.c_str(), "", f, 0,
            true);
      }
    }
  }

  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      weight_image<<<numBlocksNN, threadsPerBlockNN>>>(
          device_weight_image, datasets[d].fields[f].atten_image, N);
      checkCudaErrors(cudaDeviceSynchronize());

      if (radius_mask) {
        distance_image<<<numBlocksNN, threadsPerBlockNN>>>(
            device_distance_image, datasets[d].fields[f].ref_xobs,
            datasets[d].fields[f].ref_yobs, 4.5e-05, DELTAX, DELTAY, N);
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

  fg_scale = noise_min;
  noise_cut = noise_cut * noise_min;
  if (verbose_flag) {
    printf("fg_scale = %e\n", fg_scale);
    printf("noise (Jy/pix) = %e\n", noise_jypix);
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
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      cudaFree(datasets[d].fields[f].atten_image);
    }
  }
};

void MFS::clearRun() {
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        cudaSetDevice((i % num_gpus) + firstgpu);
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          checkCudaErrors(cudaMemset(
              datasets[d].fields[f].device_visibilities[i][s].Vr, 0,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
          checkCudaErrors(cudaMemset(
              datasets[d].fields[f].device_visibilities[i][s].Vm, 0,
              sizeof(cufftComplex) *
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        }
      }
    }
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

  if (this->gridding) {
    Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
    chi2->setCKernel(this->ckernel);
  }

  printf("\n\nStarting optimizer\n");
  if (this->Order == NULL) {
    if (imagesChanged) {
      optimizer->setImage(image);
      optimizer->optimize();
    } else if (image_count == 2) {
      optimizer->setImage(image);
      optimizer->setFlag(0);
      optimizer->optimize();
      optimizer->setFlag(1);
      optimizer->optimize();
      optimizer->setFlag(2);
      optimizer->optimize();
      optimizer->setFlag(3);
      optimizer->optimize();
    }
  } else {
    (this->Order)(optimizer, image);
  }

  float chi2_final = 0.0f;
  float final_S = 0.0f;
  float lambda_S = 0.0f;
  Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
  if (NULL != chi2) {
    chi2_final = chi2->get_fivalue();
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
  printf("Reduced-chi2 (Num visibilities): %f\n",
         chi2_final / total_visibilities);
  printf("Reduced-chi2 (Weights sum): %f\n", chi2_final / sum_weights);
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
    fprintf(outfile, "Reduced-chi2 (Num visibilities): %f\n",
            chi2_final / total_visibilities);
    fprintf(outfile, "Reduced-chi2 (Weights sum): %f\n",
            chi2_final / sum_weights);
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
                                      fg_scale, true);
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
    }
    /* code to calculate error */
    /* make void * params */
    printf("Calculating Error Images\n");
    this->error->calculateErrorImage(this->image, this->visibilities);
    if (IoOrderError == NULL) {
      if (print_images) {
        ioImageHandler->printNotNormalizedImage(
            image->getErrorImage(), "error_Inu_0.fits", "JY/PIXEL",
            optimizer->getCurrentIteration(), 0, true);
        ioImageHandler->printNotNormalizedImage(
            image->getErrorImage(), "error_alpha_0.fits", "",
            optimizer->getCurrentIteration(), 1, true);
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
    this->scheme->restoreWeights(datasets);
  } else {
    double deltax = RPDEG_D * DELTAX;  // radians
    double deltay = RPDEG_D * DELTAY;  // radians
    deltau = 1.0 / (M * deltax);
    deltav = 1.0 / (N * deltay);

    printf(
        "Visibilities are gridded, we will need to de-grid to save them in a "
        "Measurement Set File\n");
    // In the de-gridding procedure weights are also restored to the original
    for (int d = 0; d < nMeasurementSets; d++) {
      // degridding(datasets[d].fields, datasets[d].data, deltau, deltav,
      // num_gpus,
      //            firstgpu, variables.blockSizeV, M, N, this->ckernel);
      getOriginalVisibilitiesBack(datasets[d].fields, datasets[d].data,
                                  num_gpus, firstgpu, variables.blockSizeV);
    }
    Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
    chi2->setCKernel(NULL);
    chi2->simulateModel(image->getImage());
  }

  for (int d = 0; d < nMeasurementSets; d++)
    modelToHost(datasets[d].fields, datasets[d].data, num_gpus, firstgpu);

  printf("Saving residuals and model to MS...\n");
  for (int d = 0; d < nMeasurementSets; d++) {
    ioVisibilitiesHandler->copy(datasets[d].name, datasets[d].oname);
    ioVisibilitiesHandler->writeResidualsAndModel(
        datasets[d].name, datasets[d].oname, datasets[d].fields,
        datasets[d].data);
  }

  printf("Residuals and model visibilities saved.\n");
};

void MFS::unSetDevice() {
  // Free device and host memory
  printf("Freeing device memory\n");
  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        cudaSetDevice((i % num_gpus) + firstgpu);
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          cudaFree(datasets[d].fields[f].device_visibilities[i][s].uvw);
          cudaFree(datasets[d].fields[f].device_visibilities[i][s].weight);
          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vr);
          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vm);
          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vo);
        }
      }
    }
  }

  printf("Freeing cuFFT plans\n");
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    cufftDestroy(vars_gpu[g].plan);
  }

  printf("Freeing host memory\n");
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
            datasets[d].fields[f].visibilities[i][s].uvw.clear();
            datasets[d].fields[f].visibilities[i][s].weight.clear();
            datasets[d].fields[f].visibilities[i][s].Vo.clear();
            datasets[d].fields[f].visibilities[i][s].Vm.clear();

            if (this->gridding) {
              datasets[d].fields[f].backup_visibilities[i][s].uvw.clear();
              datasets[d].fields[f].backup_visibilities[i][s].weight.clear();
              datasets[d].fields[f].backup_visibilities[i][s].Vo.clear();
            }
          }
        }
      }
    }
  }

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
    free(datasets[i].name);
    free(datasets[i].oname);
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
