#include "image_processing/imageProcessor.cuh"
#include "gridding/gridder.cuh"
#include "mfs/mfs.cuh"
#include "ms/data_column.h"
#include "ms/ms_reader.h"
#include "ms/ms_writer.h"
#include "ms/polarization.h"
#include "objective_function/terms/chi2/chi2.cuh"
#include "objective_function/terms/regularizers/secondderivateerror.cuh"
#include "cli/gpuvmem_cli_config.hh"
#include "cli/cli_metrics.hh"
#include "cli/optimization_reporting.hh"
#include "cli/run_observer_factory.hh"
#include "framework.cuh"  // Still needed for many kernels and utilities
#include "main.cuh"       // print_help, goToError
#include "linesearch/linesearch_utils.cuh"  // defaultNewP, defaultEvaluateXt
#include "projection/projection_from_cli.hh"
#include "utils/physics_utils.cuh"
#include "utils/constants.hh"
#include "utils/math_utils.hh"   // iDivUp
#include "utils/cuda_utils.cuh"  // getNumBlocksAndThreads
#include "fft/fft_host.cuh"      // initFFT
#include "beam/beam_kernels.cuh" // total_attenuation, weight_image, distance_image, noise_image
#include "errors/errors_host.cuh"  // precomputeNeff
#include "framework/cuda_grid.cuh"
#include "fits/fits_io.h"
#include "io/iofits.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

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

double ra, dec, model_reference_column, model_reference_row, DELTAX, DELTAY, deltau, deltav;

std::string radesys;

float equinox;

// Per-image seed / bound values used when filling host_I (MFS / Stokes).
std::vector<float> mfs_initial_pixel_values;
// Legacy TU's (beam_host, line search / linesearch_utils) expect `float* initial_values`;
// must not reuse the symbol name for a std::vector (ODR / ABI clash → segfault).
static std::vector<float> g_legacy_initial_values_storage;
float* initial_values = nullptr;
std::vector<gpuvmem::ms::MSWithGPU> datasets;
std::vector<gpuvmem::ms::MSWithGPU>* g_datasets = nullptr;

static std::vector<float> g_penalizators_storage;
static std::vector<float> g_host_I_storage;
static std::vector<varsPerGPU> g_vars_gpu_storage;
static std::vector<imageMap> g_image_function_mapping;

varsPerGPU* vars_gpu;

bool verbose_flag, nopositivity, apply_noise, print_images, print_errors,
    save_model_input, radius_mask, modify_weights;

Vars variables;

clock_t t;
double start, end;

float noise_min = 1E32;

/** Cached for run summary (set during configure / setDevice). */
static float g_summary_freq_min_hz = 0.f;
static float g_summary_freq_max_hz = 0.f;
static double g_summary_max_uv_wavelength = 0.0;
static double g_summary_nyquist_cell_arcsec = 0.0;
static double g_summary_log_nu_min = 0.0;
static double g_summary_log_nu_max = 0.0;
static double g_summary_beam_major_arcsec = 0.0;
static double g_summary_beam_minor_arcsec = 0.0;

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

void MFS::syncLegacyGlobalsFromCli_(const GpuvmemCliConfig& cfg) {
  cli_config_ = cfg;
  gpuvmem_cli_runtime_bind(&cli_config_.runtime);
  run_observer_ = gpuvmem::cli::create_run_observer(cli_config_.runtime);
  variables = cfg.vars;
  verbose_flag = cfg.runtime.verbose;
  nopositivity = cfg.runtime.nopositivity;
  apply_noise = cfg.runtime.apply_noise;
  print_images = cfg.runtime.print_images;
  print_errors = cfg.runtime.print_errors;
  save_model_input = cfg.runtime.save_model_input;
  radius_mask = cfg.runtime.radius_mask;
  modify_weights = cfg.runtime.modify_weights;
}

void MFS::configure(const GpuvmemCliConfig& config) {
  if (ioImageHandler == NULL) {
    ioImageHandler = createObject<Io, std::string>("IoFITS");
  }

  if (ioVisibilitiesHandler == NULL) {
    ioVisibilitiesHandler = createObject<Io, std::string>("IoMS");
  }

  total_visibilities = 0;
  syncLegacyGlobalsFromCli_(config);
  ioImageHandler->setModelFitsGeometry(std::nullopt);
  msinput = variables.input;
  msoutput = variables.output;
  modinput = variables.modin;
  if (modinput == "NONE" || modinput == "-") modinput = "NULL";

  if (modinput == "NULL" || modinput.empty()) {
    if (variables.imsize.empty() || variables.cellsize_arcsec <= 0.0f ||
        variables.phase_center_deg.empty()) {
      std::cerr << "Model FITS (-m) is required unless you pass --imsize M,N "
                   "--cellsize ARCSEC --phase-center RA_DEG,DEC_DEG\n";
      print_help();
      exit(-1);
    }
    std::vector<std::string> imsz = countAndSeparateStrings(variables.imsize, ",");
    if (imsz.size() != 2) {
      std::cerr << "ERROR: --imsize must be M,N (comma-separated naxis2,naxis1)\n";
      exit(-1);
    }
    const long synM = std::strtol(imsz[0].c_str(), nullptr, 10);
    const long synN = std::strtol(imsz[1].c_str(), nullptr, 10);
    if (synM <= 0 || synN <= 0) {
      std::cerr << "ERROR: --imsize M,N requires positive integers\n";
      exit(-1);
    }
    std::vector<std::string> pc = countAndSeparateStrings(variables.phase_center_deg, ",");
    if (pc.size() != 2) {
      std::cerr << "ERROR: --phase-center must be RA_DEG,DEC_DEG\n";
      exit(-1);
    }
    const double ra_deg = std::strtod(pc[0].c_str(), nullptr);
    const double dec_deg = std::strtod(pc[1].c_str(), nullptr);
    const double cell_deg = static_cast<double>(variables.cellsize_arcsec) / 3600.0;
    gpuvmem::ImagingHeader syn{};
    syn.width_columns = synN;
    syn.height_rows = synM;
    syn.reference_column = static_cast<double>(synN / 2);
    syn.reference_row = static_cast<double>(synM / 2);
    syn.cdelt1 = -cell_deg;
    syn.cdelt2 = cell_deg;
    syn.crval1 = ra_deg;
    syn.crval2 = dec_deg;
    syn.radesys = "ICRS";
    syn.equinox = 2000.0f;
    syn.bitpix = -32;
    if (variables.path != "NULL" && !variables.path.empty()) {
      struct stat st {};
      if (stat(variables.path.c_str(), &st) != 0) {
        if (mkdir(variables.path.c_str(), 0700) != 0 && errno != EEXIST) {
          std::cerr << "ERROR: could not create output directory " << variables.path << '\n';
          exit(-1);
        }
      } else if (!S_ISDIR(st.st_mode)) {
        std::cerr << "ERROR: output path exists and is not a directory: " << variables.path << '\n';
        exit(-1);
      }
    }
    ioImageHandler->setModelFitsGeometry(syn);
    ioImageHandler->setInput("");
    modinput = "";
    if (verbose_flag)
      std::cout << "\n--- Synthetic model grid (no FITS template; WCS from CLI) ---\n"
                << "  Image size: M×N (rows×cols) = " << synM << " × " << synN << '\n'
                << "  Cell size: " << std::fixed << std::setprecision(6) << variables.cellsize_arcsec
                << " arcsec\n"
                << std::defaultfloat;
  } else if (!variables.imsize.empty() && verbose_flag) {
    std::cout << "Note: --imsize / --cellsize / --phase-center are ignored because a model FITS "
                 "(-m) is set: "
              << modinput << '\n';
  }

  if (modinput != "")
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
    std::cerr << "ERROR: no input Measurement Set(s); use -i MS1[,MS2,...]\n";
    print_help();
    exit(-1);
  }

  if (msoutput != "NULL") {
    s_output_values = countAndSeparateStrings(msoutput, ",");
    n_outputs = s_output_values.size();
  } else {
    std::cerr << "ERROR: no output path(s); use -o OUT1[,OUT2,...] (one per input MS)\n";
    print_help();
    exit(-1);
  }

  if (n_outputs != nMeasurementSets) {
    std::cerr << "ERROR: number of output MS paths (" << n_outputs << ") must match inputs ("
              << nMeasurementSets << ").\n";
    exit(-1);
  }

  if (verbose_flag)
    std::cout << "Measurement sets to read: " << nMeasurementSets << '\n';

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
    std::cerr << "ERROR: initial image level(s) not set; use -z (comma-separated per plane)\n";
    print_help();
    exit(-1);
  }

  // Store minimal_pixel_values from initial_values (before eta multiplication)
  // These will be stored in Image object
  // Store as member variable so it's accessible in setDevice() where Image object is created
  this->minimal_pixel_values.clear();
  mfs_initial_pixel_values.clear();
  for (int i = 0; i < image_count; i++) {
    if (i == 0) {
      // Set MINPIX from the first initial value (before eta multiplication)
      // This is the minimum pixel value threshold for positivity clipping
      MINPIX = std::stof(string_values[i]);
      // Store MINPIX * -1.0f * eta in minimal_pixel_values (same as initial_values)
      this->minimal_pixel_values.push_back(MINPIX * -1.0f * eta);
      // Store value after eta multiplication in initial_values
      mfs_initial_pixel_values.push_back(MINPIX * -1.0f * eta);
    } else {
      this->minimal_pixel_values.push_back(std::stof(string_values[i]));
      mfs_initial_pixel_values.push_back(std::stof(string_values[i]));
    }
  }

  string_values.clear();
  if (!variables.stokes.empty()) {
    // Stokes imaging: image_count = number of requested Stokes; validate against datasets after read
    requested_stokes = countAndSeparateStrings(variables.stokes, ",");
    image_count = static_cast<int>(requested_stokes.size());
    if (image_count <= 0) {
      std::cerr << "ERROR: --stokes must specify at least one Stokes (e.g. I or I,Q,U,V)\n";
      exit(-1);
    }
    // Ensure minimal_pixel_values and initial_values have image_count entries (pad with 0.0f)
    while (static_cast<int>(this->minimal_pixel_values.size()) < image_count)
      this->minimal_pixel_values.push_back(0.0f);
    while (static_cast<int>(mfs_initial_pixel_values.size()) < image_count)
      mfs_initial_pixel_values.push_back(0.0f);
  } else if (image_count == 1) {
    mfs_initial_pixel_values.push_back(0.0f);
    this->minimal_pixel_values.push_back(0.0f);  // Add default minimal pixel value for second image
    image_count++;
    imagesChanged = 1;
  }

  while (static_cast<int>(mfs_initial_pixel_values.size()) < image_count)
    mfs_initial_pixel_values.push_back(0.0f);
  g_legacy_initial_values_storage = mfs_initial_pixel_values;
  {
    const size_t min_len =
        std::max<size_t>(2u, static_cast<size_t>(image_count));
    if (g_legacy_initial_values_storage.size() < min_len)
      g_legacy_initial_values_storage.resize(min_len, 0.0f);
  }
  initial_values = g_legacy_initial_values_storage.data();

  /*
     Read FITS header
   */
  gpuvmem::ImagingHeader model_header = ioImageHandler->readHeader(modinput);
  // FITS: NAXIS1 = width (fast axis), NAXIS2 = height. Internal Image uses M=rows (NAXIS2),
  // N=cols (NAXIS1); phase_rotate / cuFFT use j=0..N-1, k=0..M-1 with 0-based reference_column /
  // reference_row from Image::imaging_geometry() (FITS CRPIX minus one when a header is present).
  M = model_header.height_rows;
  N = model_header.width_columns;
  ioImageHandler->setMN(M, N);
  DELTAX = model_header.cdelt1;
  DELTAY = model_header.cdelt2;
  ra = model_header.crval1;
  dec = model_header.crval2;
  ioImageHandler->setRADec(ra, dec);
  radesys = model_header.radesys;
  ioImageHandler->setFrame(radesys);
  equinox = model_header.equinox;
  cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                 std::string("\n--- Model sky frame (from FITS / template header) ---\n  RADESYS: ") +
                     radesys + "    EQUINOX: " + std::to_string(equinox) + '\n');
  ioImageHandler->setEquinox(equinox);
  model_reference_column = model_header.reference_column;
  model_reference_row = model_header.reference_row;
  if (model_header.noise_keyword > 0.0f) {
    this->setVisNoise(model_header.noise_keyword);
  }

  this->resolved_model_header_ = model_header;

  ckernel->setIoImageHandler(ioImageHandler);
  cudaGetDeviceCount(&num_gpus);
  cudaDeviceProp dprop[num_gpus];

  {
    std::ostringstream oss;
    oss << "\n--- Compute: CPUs and CUDA GPUs ---\n"
        << "  Host threads (OpenMP logical CPUs): " << omp_get_num_procs() << '\n'
        << "  CUDA devices visible: " << num_gpus << '\n';
    for (int i = 0; i < num_gpus; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&dprop[i], i));
      oss << "  GPU " << i << ": \"" << dprop[i].name << "\""
          << "    peer-to-peer capable: " << (IsGPUCapableP2P(&dprop[i]) ? "yes" : "no") << '\n';
      int memoryClockRateKHz = GetMemoryClockRateKHz(i, &dprop[i]);
      oss << "      memory clock: " << memoryClockRateKHz << " kHz"
          << "    bus width: " << dprop[i].memoryBusWidth << " bit\n";
      oss << "      peak memory bandwidth: "
          << (2.0 * memoryClockRateKHz * (dprop[i].memoryBusWidth / 8) / 1.0e6) << " GB/s"
          << "    device memory: " << (dprop[i].totalGlobalMem / pow(2, 30)) << " GiB\n";
    }
    oss << '\n';
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Debug, oss.str());
  }

  // Declaring block size and number of blocks for Image
  if (variables.blockSizeX == -1 && variables.blockSizeY == -1) {
    const gpuvmem::CudaGrid<2> grid2d_auto =
        gpuvmem::CudaGrid<2>::from_auto(M, N, dprop[0]);
    numBlocksNN = grid2d_auto.blocks();
    threadsPerBlockNN = grid2d_auto.threads();
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Debug,
                   "CUDA launch grid (image kernels): blocks (" + std::to_string(numBlocksNN.x) +
                       ", " + std::to_string(numBlocksNN.y) + "), threads per block (" +
                       std::to_string(threadsPerBlockNN.x) + ", " +
                       std::to_string(threadsPerBlockNN.y) + ")\n");
  } else {
    if (variables.blockSizeX * variables.blockSizeY >
            dprop[0].maxThreadsPerBlock) {
      std::cerr << "Block size X: " << variables.blockSizeX << '\n';
      std::cerr << "Block size Y: " << variables.blockSizeY << '\n';
      std::cerr << "Block size X*Y: " << (variables.blockSizeX * variables.blockSizeY) << '\n';
      std::cerr << "Block size V: " << variables.blockSizeV << '\n';
      std::cerr << "ERROR. The maximum threads per block cannot be greater than "
                << dprop[0].maxThreadsPerBlock << '\n';
      exit(-1);
    }
    if (variables.blockSizeV >= 0 &&
        variables.blockSizeV > dprop[0].maxThreadsPerBlock) {
      std::cerr << "Block size V: " << variables.blockSizeV << '\n';
      std::cerr << "ERROR. The maximum threads per block cannot be greater than "
                << dprop[0].maxThreadsPerBlock << '\n';
      exit(-1);
    }

    if (variables.blockSizeX > dprop[0].maxThreadsDim[0] ||
        variables.blockSizeY > dprop[0].maxThreadsDim[1]) {
      std::cerr << "Block size X: " << variables.blockSizeX << '\n';
      std::cerr << "Block size Y: " << variables.blockSizeY << '\n';
      std::cerr << "Block size V: " << variables.blockSizeV << '\n';
      std::cerr << "ERROR. The size of the blocksize cannot exceed X: " << dprop[0].maxThreadsDim[0]
                << " Y: " << dprop[0].maxThreadsDim[1] << " Z: " << dprop[0].maxThreadsDim[2]
                << '\n';
      exit(-1);
    }
    const dim3 tb(static_cast<unsigned int>(variables.blockSizeX),
                    static_cast<unsigned int>(variables.blockSizeY), 1u);
    const gpuvmem::CudaGrid<2> grid2d =
        gpuvmem::CudaGrid<2>::from_extents(M, N, tb);
    threadsPerBlockNN = grid2d.threads();
    numBlocksNN = grid2d.blocks();
  }

  if (verbose_flag)
    std::cout << "\nReading visibility data from Measurement Set(s)...\n";

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
      std::cerr << "ERROR: could not read Measurement Set: " << datasets[d].name << '\n';
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
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                   "  MS [" + std::to_string(d) + "] " + datasets[d].name +
                       "  antenna_diam_m=" + std::to_string(ant_diam) + '\n');
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
        std::cerr << "ERROR: dataset " << d << " (" << datasets[d].name
                  << ") cannot form requested Stokes (";
        for (size_t i = 0; i < requested_stokes.size(); i++) {
          std::cerr << requested_stokes[i];
          if (i + 1 < requested_stokes.size()) std::cerr << ',';
        }
        std::cerr << "). Available from correlations: ";
        for (size_t i = 0; i < available.size(); i++) {
          std::cerr << available[i];
          if (i + 1 < available.size()) std::cerr << ',';
        }
        std::cerr << '\n';
        exit(-1);
      }
    }
    if (verbose_flag)
      std::cout << "Stokes planes to solve for: " << image_count << " (" << variables.stokes
                << ")\n";
    // Convert visibility data to Stokes so pol index = Stokes index (I=0, Q=1, U=2, V=3)
    for (int d = 0; d < nMeasurementSets; d++) {
      if (!gpuvmem::ms::correlations_to_stokes(datasets[d].ms,
                                                requested_stokes)) {
        std::cerr << "ERROR: failed to convert dataset " << d << " (" << datasets[d].name
                  << ") to Stokes\n";
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
  g_summary_freq_min_hz = min_freq;
  g_summary_freq_max_hz = max_freq;
  g_summary_max_uv_wavelength = max_uvmax_wavelength;

  if (nu_0 <= 0.0f) {
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                   "WARNING: reference frequency nu_0 not set (or nu_0<=0); using band centre.\n");
    nu_0 = 0.5f * (max_freq + min_freq);
  }
  g_summary_log_nu_min = log(static_cast<double>(min_freq) / static_cast<double>(nu_0));
  g_summary_log_nu_max = log(static_cast<double>(max_freq) / static_cast<double>(nu_0));
  if (fabs(g_summary_log_nu_max - g_summary_log_nu_min) < 0.01 && image_count > 1) {
    cli_diagnostic(
        gpuvmem::cli::DiagnosticLevel::Verbose,
        "WARNING: log(nu/nu_0) range is very narrow; spectral index alpha will be weakly "
        "constrained (errors may be large or hit limits).\n");
  }
  {
    double deltau_theo = 2.0 * max_uvmax_wavelength / (M - 1);
    g_summary_nyquist_cell_arcsec = 1.0 / (M * deltau_theo) / RPARCSEC;
  }
  (void)resolution_arcsec;

  if (verbose_flag) {
    std::cout << "\n--- Measurement Set metadata (verbose) ---\n";
    for (int i = 0; i < nMeasurementSets; i++) {
      size_t nchan = 0;
      int npol = 0;
      for (const auto& dd : datasets[i].ms.metadata().data_descriptions()) {
        nchan += dd.nchan();
        if (dd.npol() > 0) npol = dd.npol();
      }
      std::cout << "  MS [" << i << "] " << datasets[i].name << '\n'
                << "      fields: " << datasets[i].ms.num_fields() << '\n'
                << "      spectral channels (sum over SPWs): " << nchan << '\n'
                << "      correlations per row: " << npol << '\n';
    }
  }

  multigpu = 0;
  firstgpu = 0;
  int count_gpus;

  if (variables.gpus == "NULL" || variables.gpus.empty()) {
    string_values.clear();
  } else {
    string_values = countAndSeparateStrings(variables.gpus, ",");
  }
  count_gpus = static_cast<int>(string_values.size());

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
  if (variables.regularization_weights != "NULL") {
    string_values =
        countAndSeparateStrings(variables.regularization_weights, ",");
    nPenalizators = static_cast<int>(string_values.size());
    g_penalizators_storage.resize(static_cast<size_t>(nPenalizators));
    for (int i = 0; i < nPenalizators; i++) {
      g_penalizators_storage[static_cast<size_t>(i)] = std::stof(string_values[i]);
    }
    penalizators =
        g_penalizators_storage.empty() ? nullptr : g_penalizators_storage.data();

  } else {
    std::cout << "\nNote: no -y regularization weights; only configured objective terms apply.\n";
    nPenalizators = 0;
    g_penalizators_storage.clear();
    penalizators = nullptr;
  }
  string_values.clear();

  int max_nfreq = 1;
  if (multigpu < 0 || multigpu > num_gpus) {
    std::cerr << "ERROR: GPU count (-G) must be between 0 and the number of visible CUDA devices.\n";
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
        std::cout << "Note: only one spectral channel in the data; using a single GPU for "
                     "frequency partitioning.\n";
        num_gpus = 1;
      } else {
        num_gpus = multigpu;
      }
    }
  }

  int total_gpus;
  cudaGetDeviceCount(&total_gpus);
  if (firstgpu > total_gpus - 1 || firstgpu < 0) {
    std::cerr << "ERROR: requested primary GPU index is out of range for this machine.\n";
    exit(-1);
  }

  if (verbose_flag) {
    std::cout << "GPUs used for this run: " << num_gpus << '\n';
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
        std::cout << "  Peer access GPU " << firstgpu << " -> GPU " << i << ": "
                  << (canAccessPeer0_x ? "yes" : "no") << '\n';
        std::cout << "  Peer access GPU " << i << " -> GPU " << firstgpu << ": "
                  << (canAccessPeerx_0 ? "yes" : "no") << '\n';
      }
      if (canAccessPeer0_x == 0 || canAccessPeerx_0 == 0) {
        std::cout << "Multi-GPU run requested (" << num_gpus << " devices) but peer access is "
                     "missing between GPU "
                  << firstgpu << " and GPU " << i << ".\n";
        std::cout << "This build expects bidirectional P2P and unified addressing for multi-GPU.\n";
        std::cout << "Exiting (see CUDA multi-GPU / NVLink requirements).\n";
        exit(EXIT_SUCCESS);
      } else {
        cudaSetDevice(firstgpu);
        if (verbose_flag) {
          std::cout << "  Enabling peer access: GPU " << firstgpu << " -> GPU " << i << "...\n";
        }
        cudaDeviceEnablePeerAccess(i, 0);
        cudaSetDevice(i);
        if (verbose_flag) {
          std::cout << "  Enabling peer access: GPU " << i << " -> GPU " << firstgpu << "...\n";
        }
        cudaDeviceEnablePeerAccess(firstgpu, 0);
        if (verbose_flag) {
          std::cout << "  Checking unified virtual addressing (UVA) on GPU " << firstgpu
                    << " and GPU " << i << "...\n";
        }
        const bool has_uva =
            (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
        if (verbose_flag) {
          std::cout << "    GPU " << firstgpu << " (\"" << dprop0.name << "\") UVA: "
                    << (dprop0.unifiedAddressing ? "yes" : "no") << '\n';
          std::cout << "    GPU " << i << " (\"" << dpropX.name << "\") UVA: "
                    << (dpropX.unifiedAddressing ? "yes" : "no") << '\n';
        }
        if (has_uva) {
          if (verbose_flag) {
            std::cout << "  Both devices support UVA; continuing.\n";
          }
        } else {
          std::cout << "At least one GPU lacks UVA; multi-GPU path is not supported here.\n";
          exit(EXIT_SUCCESS);
        }
      }
    }
  }

  g_vars_gpu_storage.assign(static_cast<size_t>(num_gpus), varsPerGPU{});
  vars_gpu = g_vars_gpu_storage.data();

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
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                   "\n--- Visibility gridding ---\nGridding visibilities onto the UV grid...\n");
    this->ckernel->setSigmas(fabs(deltau), fabs(deltav));
    this->ckernel->buildKernel();
    if (gpuvmem_cli_debug()) {
      this->ckernel->printCKernel();
    }
    this->ckernel->initializeGCF(M, N, fabs(deltax), fabs(deltay));
    if (gpuvmem_cli_debug()) {
      this->ckernel->printGCF();
    }
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                   std::string("Convolution kernel: ") + this->ckernel->getName() +
                       "  kernel size (u,v): (" +
            std::to_string(this->ckernel->getm()) + ", " + std::to_string(this->ckernel->getn()) +
            ")  support: (" + std::to_string(this->ckernel->getSupportX()) + ", " +
            std::to_string(this->ckernel->getSupportY()) + ")\n");
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
    std::cout << "Measurement Set I/O finished (read";
    if (this->gridding) {
      std::cout << " + gridding to the UV grid";
    }
    std::cout << ").\n";
    if (this->getVisNoise() < 0.0f) {
      std::cout << "Visibility noise not set; estimating from weights / UV coverage...\n";
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
    // Chi² gather uses nch = number of chunks with a given (chan, pol), which can
    // approach total_chunk_count. total_visibilities() is sum of chunk sizes; use both.
    const size_t tv = datasets[d].gpu.total_visibilities();
    const size_t tc = datasets[d].gpu.total_chunk_count();
    const size_t need = std::max(tv, tc);
    const int cap = need > static_cast<size_t>(std::numeric_limits<int>::max())
                        ? std::numeric_limits<int>::max()
                        : static_cast<int>(need);
    if (cap > max_number_vis) max_number_vis = cap;
  }

  if (max_number_vis == 0) {
    std::cerr << "ERROR: no usable visibilities after setup (max visibilities = 0). Check inputs "
                 "and flags.\n";
    exit(-1);
  }

  this->setMaxNumberVis(max_number_vis);

  g_summary_beam_major_arcsec = beam_bmaj * 3600.0;
  g_summary_beam_minor_arcsec = beam_bmin * 3600.0;
  cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                 std::string("\n--- Noise and synthesized beam (from weights / UV) ---\n") +
                     "  Estimated restoring beam FWHM: " +
                     std::to_string(g_summary_beam_major_arcsec) + " x " +
                     std::to_string(g_summary_beam_minor_arcsec) + " arcsec, PA " +
                     std::to_string(beam_bpa) + " deg\n");
  beam_bmaj = beam_bmaj / fabs(DELTAX);  // Beam major axis to pixels
  beam_bmin = beam_bmin / fabs(DELTAX);  // Beam minor axis to pixels
  noise_jypix =
      this->getVisNoise() / (PI_D * beam_bmaj * beam_bmin /
                             (4.0 * logf(2.0)));  // Estimating noise at FWHM

  /////////////////////////////////////////////////////CALCULATE DIRECTION
  /// COSINES/////////////////////////////////////////////////
  cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                 "\nAligning MS pointing with model WCS (reference / phase centres vs CRPIX)...\n");
  double raimage = ra * RPDEG_D;
  double decimage = dec * RPDEG_D;

  if (verbose_flag) {
    std::cout << "\n--- Model phase centre (from FITS CRVAL; radians) ---\n";
    std::cout << std::scientific << std::setprecision(16) << "  RA: " << raimage << " rad    Dec: "
              << decimage << " rad\n"
              << std::defaultfloat;
    std::cout << "  Reference pixel (0-based column, row; aligns with MS l,m): ("
              << model_reference_column << ", " << model_reference_row << ")\n";
  }

  double lobs, mobs, lphs, mphs;
  double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs,
      dcosines_m_pix_phs;
  for (int d = 0; d < nMeasurementSets; d++) {
    if (verbose_flag) std::cout << "\n  MS: " << datasets[d].name << '\n';
    gpuvmem::ms::MeasurementSet& ms = datasets[d].ms;
    for (size_t f = 0; f < ms.num_fields(); f++) {
      gpuvmem::ms::Field& field = ms.field(f);
      gpuvmem::ms::FieldMetadata& fmeta = field.metadata();
      direccos(field.reference_dir()[0], field.reference_dir()[1],
               raimage, decimage, &lobs, &mobs);
      dcosines_l_pix_ref = lobs / deltax;
      dcosines_m_pix_ref = mobs / deltay;
      direccos(field.phase_dir()[0], field.phase_dir()[1],
               raimage, decimage, &lphs, &mphs);

      dcosines_l_pix_phs = lphs / deltax;  // Radians to pixels
      dcosines_m_pix_phs = mphs / deltay;  // Radians to pixels

      if (verbose_flag) {
        std::cout << std::scientific << "    Field " << f << " : offset of reference direction in "
                                      "image plane: l = "
                  << dcosines_l_pix_ref << " pix, m = " << dcosines_m_pix_ref << " pix\n";
        std::cout << "    Field " << f << " : offset of phase centre in image plane: l = "
                  << dcosines_l_pix_phs << " pix, m = " << dcosines_m_pix_phs << " pix\n"
                  << std::defaultfloat;
      }
      fmeta.ref_xobs_pix = static_cast<float>(dcosines_l_pix_phs + model_reference_column);
      fmeta.ref_yobs_pix = static_cast<float>(dcosines_m_pix_phs + model_reference_row);
      fmeta.phs_xobs_pix = static_cast<float>(dcosines_l_pix_phs + model_reference_column);
      fmeta.phs_yobs_pix = static_cast<float>(dcosines_m_pix_phs + model_reference_row);

      if (verbose_flag) {
        std::cout << std::scientific << std::setprecision(16)
                  << "    Field " << f << " reference direction: RA = " << field.reference_dir()[0]
                  << " rad, Dec = " << field.reference_dir()[1]
                  << " rad\n      mapped pixel (col, row): (" << std::defaultfloat << fmeta.ref_xobs_pix
                  << ", " << fmeta.ref_yobs_pix << ")\n";
        std::cout << std::scientific << std::setprecision(16)
                  << "    Field " << f << " phase centre: RA = " << field.phase_dir()[0]
                  << " rad, Dec = " << field.phase_dir()[1]
                  << " rad\n      mapped pixel (col, row): (" << std::defaultfloat << fmeta.phs_xobs_pix
                  << ", " << fmeta.phs_yobs_pix << ")\n";
      }

      if (fmeta.ref_xobs_pix < 0 || fmeta.ref_xobs_pix >= N ||
          fmeta.ref_yobs_pix < 0 || fmeta.ref_yobs_pix >= M) {
        std::cerr << "ERROR: MS \"" << datasets[d].name << "\"\n";
        std::cerr << "  Reference centre falls outside the image grid at pixel (col,row) = ("
                  << fmeta.ref_xobs_pix << ',' << fmeta.ref_yobs_pix << ").\n";
        goToError();
      }

      if (fmeta.phs_xobs_pix < 0 || fmeta.phs_xobs_pix >= N ||
          fmeta.phs_yobs_pix < 0 || fmeta.phs_yobs_pix >= M) {
        std::cerr << "ERROR: MS \"" << datasets[d].name << "\"\n";
        std::cerr << "  Phase centre falls outside the image grid at pixel (col,row) = ("
                  << fmeta.phs_xobs_pix << ',' << fmeta.phs_yobs_pix << ").\n";
        goToError();
      }
    }
  }
  // Gridder::grid() snapshots FieldMetadata into gridded_ms before this block runs;
  // copy the filled ref_/phs_ centers so synthesis uses the same pixels as the native MS.
  for (int d = 0; d < nMeasurementSets; ++d) {
    gpuvmem::ms::MSWithGPU& dw = datasets[d];
    if (dw.gridded_ms.num_fields() == 0) continue;
    const size_t n =
        std::min(dw.ms.num_fields(), dw.gridded_ms.num_fields());
    for (size_t f = 0; f < n; ++f) {
      dw.gridded_ms.field(f).metadata() = dw.ms.field(f).metadata();
    }
  }
  ////////////////////////////////////////////////////////MAKE STARTING
  /// IMAGE////////////////////////////////////////////////////////

  const size_t host_pixels =
      static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(image_count);
  g_host_I_storage.resize(host_pixels);
  host_I = g_host_I_storage.data();

  for (int k = 0; k < image_count; k++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        host_I[N * M * k + N * i + j] = mfs_initial_pixel_values[k];
      }
    }
  }

  // Optional: replace plane 0 (e.g. I_nu_0 or first Stokes) from FITS; -z still sets MINPIX,
  // minimal_pixel_values, and constant fills for planes k>=1 (e.g. spectral index).
  if (variables.initial_model_fits != "NULL" && !variables.initial_model_fits.empty()) {
    checkCudaErrors(cudaSetDevice(firstgpu));
    Image* init_img = nullptr;
    try {
      init_img = IoFITS::readImageFromFits(variables.initial_model_fits, firstgpu);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: --initial-model " << variables.initial_model_fits << ": " << e.what()
                << '\n';
      goToError();
    }
    if (init_img->getM() != M || init_img->getN() != N) {
      std::cerr << "ERROR: --initial-model grid (" << init_img->getM() << ',' << init_img->getN()
                << ") does not match reconstruction grid (" << M << ',' << N << ")\n";
      checkCudaErrors(cudaFree(init_img->getImage()));
      delete init_img;
      goToError();
    }
    if (init_img->getImageCount() != 1) {
      std::cerr << "ERROR: internal: initial FITS image must be single-plane\n";
      checkCudaErrors(cudaFree(init_img->getImage()));
      delete init_img;
      goToError();
    }
    const size_t plane_bytes = static_cast<size_t>(M * N) * sizeof(float);
    checkCudaErrors(cudaMemcpy(host_I, init_img->getImage(), plane_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(init_img->getImage()));
    delete init_img;
    if (verbose_flag) {
      std::cout << "Initial image plane 0 loaded from FITS (--initial-model): "
                << variables.initial_model_fits << '\n';
    }
    if (image_count > 1 && verbose_flag) {
      std::cout << "Note: planes 1.." << (image_count - 1)
                << " remain at the constant level from -z (not read from FITS).\n";
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
  image->setImagingHeader(this->resolved_model_header_);
  // Set minimal pixel values from initial_values (before eta multiplication)
  // minimal_pixel_values was declared in configure() and stored as member variable
  image->setMinimalPixelValues(this->minimal_pixel_values);
  if (this->optimizer != nullptr) {
    this->optimizer->setProjection(makeLineSearchProjectionFromCli(
        this->cli_config_, g_legacy_initial_values_storage, this->minimal_pixel_values));
  }
  // Set global I pointer for backward compatibility (used as fallback in line searchers)
  // Note: Line searchers prefer using this->image from optimizer, but fall back to extern I if needed
  I = image;
  // Chi2::configureImage is called from MFS::run() after main adds Chi2 to the
  // objective function; getFiByName here would still be null during setDevice().
  g_image_function_mapping.resize(static_cast<size_t>(image_count));
  for (int i = 0; i < image_count; i++) {
    g_image_function_mapping[static_cast<size_t>(i)].evaluateXt = defaultEvaluateXt;
    g_image_function_mapping[static_cast<size_t>(i)].newP = defaultNewP;
  }
  image->setFunctionMapping(g_image_function_mapping.data());

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

  std::vector<float> host_weight_image(static_cast<size_t>(M) * static_cast<size_t>(N));
  checkCudaErrors(cudaMemcpy2D(host_weight_image.data(), sizeof(float),
                               device_weight_image, sizeof(float),
                               sizeof(float), M * N, cudaMemcpyDeviceToHost));
  float max_weight =
      *std::max_element(host_weight_image.begin(), host_weight_image.end());

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

  std::vector<float> host_noise_image(static_cast<size_t>(M) * static_cast<size_t>(N));
  checkCudaErrors(cudaMemcpy2D(host_noise_image.data(), sizeof(float),
                               device_noise_image, sizeof(float), sizeof(float),
                               M * N, cudaMemcpyDeviceToHost));
  float noise_min =
      *std::min_element(host_noise_image.begin(), host_noise_image.end());
  if (!std::isfinite(noise_min) || noise_min <= 0.0f) {
    if (verbose_flag) {
      std::cerr << std::scientific << "WARNING: invalid minimum in derived noise map (" << noise_min
                << "); using fg_scale = 1.0 for scaling.\n"
                << std::defaultfloat;
    }
    noise_min = 1.0f;
  }

  this->fg_scale = noise_min;
  noise_cut = noise_cut * noise_min;
  // MINPIX is set from the -z option (first initial value) at line 181
  // Note: 0.0 is a valid value for MINPIX (user may want values >= 0)
  // So we don't check if MINPIX == 0.0f, as that would incorrectly overwrite
  // a valid user-specified value of 0.0
  if (verbose_flag) {
    std::cout << "\n--- Image-domain scaling (noise / masks) ---\n"
              << std::scientific << "  fg_scale (noise floor factor): " << this->fg_scale << '\n'
              << "  Estimated map noise: " << noise_jypix << " Jy/pixel\n"
              << std::defaultfloat << "  MINPIX (from -z, first plane): " << MINPIX << '\n';
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

void MFS::cli_diagnostic(gpuvmem::cli::DiagnosticLevel level,
                         const std::string& message) const {
  if (run_observer_ != nullptr) {
    run_observer_->on_diagnostic(level, message);
    return;
  }
  if (level == gpuvmem::cli::DiagnosticLevel::Debug && !gpuvmem_cli_debug()) return;
  if (level == gpuvmem::cli::DiagnosticLevel::Verbose && !gpuvmem_cli_verbose()) return;
  std::cout << message;
  if (!message.empty() && message.back() != '\n') std::cout << '\n';
}

gpuvmem::cli::RunSummary MFS::build_run_summary() const {
  gpuvmem::cli::RunSummary s;
  s.n_ms = nMeasurementSets;
  for (int i = 0; i < nMeasurementSets; ++i) {
    s.input_ms.push_back(datasets[i].name);
    s.output_ms.push_back(datasets[i].oname);
  }
  if (modinput != "NULL" && !modinput.empty()) {
    s.model_description = std::string("FITS ") + modinput;
  } else {
    s.model_description = "synthetic imsize=" + variables.imsize + " cell=" +
                          std::to_string(variables.cellsize_arcsec) + " arcsec phase=" +
                          variables.phase_center_deg;
  }
  s.radesys = radesys;
  s.equinox = static_cast<double>(equinox);
  s.phase_ra_deg = ra;
  s.phase_dec_deg = dec;
  s.grid_m = M;
  s.grid_n = N;
  s.cell_arcsec = fabs(DELTAX) * 3600.0;
  s.ref_col = model_reference_column;
  s.ref_row = model_reference_row;
  if (!variables.stokes.empty()) {
    s.planes_description =
        std::to_string(image_count) + "  Stokes: " + variables.stokes;
  } else if (image_count == 2) {
    s.planes_description = "2  labels=I_nu_0,alpha";
  } else if (image_count == 1) {
    s.planes_description = "1  single_plane";
  } else {
    s.planes_description = std::to_string(image_count) + "  image_planes";
  }
  s.freq_min_hz = g_summary_freq_min_hz;
  s.freq_max_hz = g_summary_freq_max_hz;
  s.nu0_hz = nu_0;
  s.log_nu_min = g_summary_log_nu_min;
  s.log_nu_max = g_summary_log_nu_max;
  s.vis_rows_used = total_visibilities;
  s.random_sample_frac = variables.randoms;
  s.gridding = this->gridding;
  s.weighting = variables.weighting_scheme;
  s.robust_r = variables.robust_param;
  s.max_uv_wavelength = g_summary_max_uv_wavelength;
  s.nyquist_cell_arcsec = g_summary_nyquist_cell_arcsec;
  s.beam_major_arcsec = g_summary_beam_major_arcsec;
  s.beam_minor_arcsec = g_summary_beam_minor_arcsec;
  s.beam_pa_deg = beam_bpa;
  s.noise_jy_per_pixel = noise_jypix;
  s.fg_scale = this->fg_scale;
  static const char* k_fi_names[] = {"Chi2", "Entropy", "L1-Norm", "TotalSquaredVariation"};
  ObjectiveFunction* of =
      optimizer ? optimizer->getObjectiveFunction() : nullptr;
  for (const char* name : k_fi_names) {
    gpuvmem::cli::ObjectiveTermLine line;
    line.name = name;
    Fi* fi = (of != nullptr) ? of->getFiByName(name) : nullptr;
    line.active = (fi != nullptr);
    line.lambda = (fi != nullptr) ? static_cast<double>(fi->getPenalizationFactor()) : 0.0;
    s.objective_terms.push_back(line);
  }
  s.optimizer_id = variables.optimizer_name;
  s.optimization_mode = variables.optimization_mode;
  s.max_iter = variables.it_max;
  if (optimizer != nullptr) {
    s.ftol = static_cast<double>(optimizer->getFtol());
    s.gtol = static_cast<double>(optimizer->getGtol());
  }
  s.linesearch_id = variables.linesearch_name.empty() ? "Brent" : variables.linesearch_name;
  s.seeder_id = variables.seeder_name.empty() ? "none" : variables.seeder_name;
  s.lbfgs_m = variables.lbfgs_corrections;
  Fi* chi2 = (of != nullptr) ? of->getFiByName("Chi2") : nullptr;
  s.normalize_chi2 = (chi2 != nullptr && chi2->getNormalize());
  s.minpix = MINPIX;
  s.noise_cut_eff = noise_cut;
  if (variables.user_mask != "NULL" && !variables.user_mask.empty()) {
    s.mask_description = "user:" + variables.user_mask;
  } else if (radius_mask) {
    s.mask_description = "radius";
  } else {
    s.mask_description = "noise";
  }
  s.gpu_first = firstgpu;
  s.gpu_count = num_gpus;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, firstgpu) == cudaSuccess) {
    s.gpu_name = prop.name;
    s.gpu_mem_gib = prop.totalGlobalMem / std::pow(2.0, 30.0);
  }
  return s;
}

void MFS::emit_run_summary() const {
  if (run_observer_ == nullptr) return;
  run_observer_->on_run_summary(build_run_summary());
}

void MFS::run() {
  optimizer->getObjectiveFunction()->setIo(ioImageHandler);
  optimizer->getObjectiveFunction()->setPrimaryCudaDevice(firstgpu);

  Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
  // Use static_cast: CUDA builds often compile without RTTI, so dynamic_cast
  // would fail and skip configureImage (Chi2 is the concrete Fi for "Chi2").
  if (chi2 != nullptr && image != nullptr)
    static_cast<Chi2*>(chi2)->configureImage(image);

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
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose,
                   "Pre-computing effective sample count N_eff for normalized chi^2...\n");
    precomputeNeff(true);
  }

  optimizer->setRunObserver(run_observer_.get());
  emit_run_summary();

  gpuvmem::cli::OptimizationBeginInfo begin_info;
  begin_info.optimizer_id = variables.optimizer_name;
  begin_info.max_iterations = variables.it_max;
  const std::string& mode = variables.optimization_mode;

  auto start_optimization = [&](const std::string& mode_desc, int sub_run, int sub_total,
                                const std::string& plane) {
    begin_info.mode_description = mode_desc;
    if (run_observer_ != nullptr) {
      run_observer_->on_optimization_begin(begin_info);
      run_observer_->reset_iteration_progress();
    }
    optimizer->setOptimizationSubRunContext(sub_run, sub_total, plane);
  };

  if (image_count == 1) {
    start_optimization("Mode: single image plane (Stokes or one-parameter reconstruction).", 0, 0,
                       "");
    optimizer->setImage(image);
    optimizer->setFlag(0);
    optimizer->optimize();
  } else if (image_count == 2) {
    optimizer->setImage(image);
    if (mode == "joint") {
      start_optimization("Mode: joint — I(nu_0) and spectral index alpha updated together.", 0, 0,
                         "");
      optimizer->setFlag(-1);
      optimizer->optimize();
    } else if (mode == "alpha_static") {
      start_optimization("Mode: alpha fixed — only I(nu_0); spectral index held constant.", 0, 0,
                         "");
      optimizer->setFlag(0);
      optimizer->optimize();
    } else if (mode == "block" && this->Order != NULL) {
      start_optimization(
          "Mode: alternating blocks — I(nu_0) and alpha in separate sub-problems.", 0, 0, "");
      (this->Order)(optimizer, image);
    } else {
      start_optimization(
          "Mode: alternating blocks — I(nu_0) and alpha in separate sub-problems.", 0, 0, "");
      optimizer->setOptimizationSubRunContext(1, 4, "I_nu_0");
      optimizer->setFlag(0);
      optimizer->optimize();
      optimizer->setOptimizationSubRunContext(2, 4, "alpha");
      optimizer->setFlag(1);
      optimizer->optimize();
      optimizer->setOptimizationSubRunContext(3, 4, "block_2");
      optimizer->setFlag(2);
      optimizer->optimize();
      optimizer->setOptimizationSubRunContext(4, 4, "block_3");
      optimizer->setFlag(3);
      optimizer->optimize();
    }
  } else if (this->Order != NULL) {
    start_optimization("Mode: custom multi-plane order.", 0, 0, "");
    (this->Order)(optimizer, image);
  } else if (imagesChanged) {
    start_optimization("Mode: auxiliary image plane.", 0, 0, "");
    optimizer->setImage(image);
    optimizer->optimize();
  }

  t = clock() - t;
  end = omp_get_wtime();
  const double cpu_time_s = static_cast<double>(t) / CLOCKS_PER_SEC;
  const double wall_time_s = end - start;

  if (verbose_flag) {
    std::ostringstream oss;
    oss << "--- Run configuration (verbose) ---\n"
        << "  Grid: M(rows)=" << M << "  N(cols)=" << N << "  image planes=" << image_count
        << "  primary GPU=" << firstgpu << '\n'
        << "  Weighting: " << variables.weighting_scheme << "  Optimizer: "
        << variables.optimizer_name << "  Mode: " << variables.optimization_mode << '\n'
        << "  Line search: \"" << variables.linesearch_name << "\"  Seeder: \""
        << variables.seeder_name << "\"  L-BFGS history M=" << variables.lbfgs_corrections << '\n';
    cli_diagnostic(gpuvmem::cli::DiagnosticLevel::Verbose, oss.str());
  }

  ObjectiveFunction* of = optimizer->getObjectiveFunction();
  const gpuvmem::cli::FinalRunMetrics final_metrics = gpuvmem::cli::build_final_run_metrics(
      optimizer, of, image->getImage(), variables.optimizer_name, variables.optimization_mode,
      variables.it_max, cpu_time_s, wall_time_s);

  if (run_observer_ != nullptr) {
    run_observer_->on_final_metrics(final_metrics);
    std::string out_path;
    if (variables.output_image != "NULL" && !variables.output_image.empty()) {
      out_path = variables.output_image;
    } else if (variables.path != "NULL" && !variables.path.empty()) {
      out_path = variables.path;
    }
    run_observer_->on_run_finished_quiet(final_metrics.iteration_last, final_metrics.iteration_budget,
                                         static_cast<double>(final_metrics.phi_total), wall_time_s,
                                         out_path);
  }

  if (variables.ofile != "NULL" && !variables.ofile.empty()) {
    std::ofstream outfile(variables.ofile);
    if (!outfile) {
      std::cerr << "ERROR: could not open metrics file for writing: " << variables.ofile << '\n';
      goToError();
    }
    gpuvmem::cli::write_final_metrics(outfile, final_metrics);
  }
};

void MFS::writeImages() {
  std::cout << "\nWriting reconstructed image(s) to disk...\n";
  
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
    std::cout << "Computing error / uncertainty images...\n";
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
            std::snprintf(fname, sizeof(fname), "error_stokes_%d.fits", p);
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
  std::cout << "Copying residual visibilities to host memory...\n";
  if (!this->gridding) {
    this->scheme->restoreWeights(*this->getDatasets());
  } else {
    double deltax = RPDEG_D * DELTAX;  // radians
    double deltay = RPDEG_D * DELTAY;  // radians
    deltau = 1.0 / (M * deltax);
    deltav = 1.0 / (N * deltay);

    std::cout << "Data were gridded; de-gridding to native UV locations before writing the MS...\n";
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
      gridder.degrid(datasets[d], image->getImage(), ip, image);
      datasets[d].download();
    }
    delete ip;

    int cap = max_number_vis;
    for (int d = 0; d < nMeasurementSets; d++) {
      const size_t tv = datasets[d].gpu.total_visibilities();
      const size_t tc = datasets[d].gpu.total_chunk_count();
      const size_t need = std::max(tv, tc);
      const int ni = need > static_cast<size_t>(std::numeric_limits<int>::max())
                         ? std::numeric_limits<int>::max()
                         : static_cast<int>(need);
      if (ni > cap) cap = ni;
    }
    if (cap > max_number_vis) {
      max_number_vis = cap;
      for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(firstgpu + g);
        cudaFree(vars_gpu[g].device_chi2);
        checkCudaErrors(cudaMalloc(&vars_gpu[g].device_chi2,
                                   sizeof(float) * max_number_vis));
      }
    }

    Fi* chi2 = optimizer->getObjectiveFunction()->getFiByName("Chi2");
    float res = chi2->calcFi(image->getImage());
    std::cout << "Chi^2 on the original (non-gridded) UV grid after de-gridding: " << res << '\n';
  }

  for (int d = 0; d < nMeasurementSets; d++) {
    datasets[d].download();
  }
  std::cout << "Writing MODEL_DATA and residuals to output Measurement Set(s)...\n";
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
      std::cerr << "ERROR: could not write MS \"" << datasets[d].oname << "\" (MSWriter failed).\n";
    }
  }
  std::cout << "Residual and model columns written successfully.\n";
};

void MFS::unSetDevice() {
  std::cout << "Releasing GPU memory...\n";
  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    datasets[d].clear();
    for (size_t f = 0; f < datasets[d].atten_image.size(); f++) {
      cudaFree(datasets[d].atten_image[f]);
    }
    datasets[d].atten_image.clear();
  }

  std::cout << "Destroying cuFFT plans...\n";
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    cufftDestroy(vars_gpu[g].plan);
  }

  std::cout << "Releasing host buffers...\n";
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
  g_host_I_storage.clear();
  host_I = nullptr;
  g_penalizators_storage.clear();
  penalizators = nullptr;
  nPenalizators = 0;
  g_vars_gpu_storage.clear();
  vars_gpu = nullptr;
  g_image_function_mapping.clear();

  for (int i = 0; i < nMeasurementSets; i++) {
    datasets[i].name.clear();
    datasets[i].oname.clear();
  }
}

namespace {
Synthesizer* CreateMFS() {
  return new MFS;
}
const std::string name = "MFS";
const bool RegisteredMFS =
    registerCreationFunction<Synthesizer, std::string>(name, CreateMFS);
};  // namespace
