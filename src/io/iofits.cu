#include "io/iofits.cuh"
#include "fits/fits_io.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

/** Build WriteFitsImageOptions from IoFITS state and write one slice; exits on error. */
void write_slice(float* I, const char* path, const char* name_image,
                 const char* units, int iteration, int index, float fg_scale,
                 long M, long N, double ra_center, double dec_center,
                 std::string frame, float equinox, bool isInGPU,
                 const std::string& template_path) {
  gpuvmem::fits::WriteFitsImageOptions opts;
  opts.header_template = template_path;
  opts.output_path = (path && path[0]) ? (std::string(path) + name_image) : name_image;
  opts.data = I;
  opts.naxis1 = M;
  opts.naxis2 = N;
  opts.plane_index = index;
  opts.bunit = units ? units : "";
  opts.niter = iteration;
  opts.normalization_factor = fg_scale;
  opts.crval1 = ra_center;
  opts.crval2 = dec_center;
  opts.radesys = frame.empty() ? "ICRS" : frame;
  opts.equinox = equinox;
  opts.normalize = (fg_scale != 1.0f);
  opts.data_on_device = isInGPU;
  try {
    gpuvmem::fits::write_fits_image_slice(opts);
  } catch (const std::exception& e) {
    fprintf(stderr,
            "FITS output failed (writes clone the header from the model FITS; "
            "see message for read vs write): %s\n",
            e.what());
    std::exit(1);
  }
}

/** Build WriteFitsComplexImageOptions and write complex image; exits on error. */
void write_complex_slice(cufftComplex* I, const char* template_filename,
                         const char* path, const char* out_image, int iteration,
                         long M, long N, int option, bool isInGPU) {
  gpuvmem::fits::WriteFitsComplexImageOptions opts;
  opts.header_template = template_filename ? template_filename : "";
  opts.data = I;
  opts.naxis1 = M;
  opts.naxis2 = N;
  opts.niter = iteration;
  opts.output_type = gpuvmem::fits::WriteFitsComplexImageOptions::AMPLITUDE;
  opts.data_on_device = isInGPU;
  
  // Build output path based on option (matching OCopyFITSCufftComplex behavior)
  std::string output_path;
  switch (option) {
    case 0:
      output_path = out_image ? out_image : "";
      break;
    case 1:
      if (path && path[0]) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%sMEM_%d.fits", path, iteration);
        output_path = buf;
      } else {
        char buf[256];
        snprintf(buf, sizeof(buf), "MEM_%d.fits", iteration);
        output_path = buf;
      }
      break;
    case -1:
      fprintf(stderr, "Invalid case to FITS\n");
      std::exit(-1);
      break;
    default:
      fprintf(stderr, "Invalid case to FITS\n");
      std::exit(-1);
      break;
  }
  opts.output_path = output_path;
  
  try {
    gpuvmem::fits::write_fits_image_complex(opts);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS complex write failed: %s\n", e.what());
    std::exit(1);
  }
}

IoFITS::IoFITS() : Io() {
  this->M = 0;
  this->N = 0;
  this->normalization_factor = 1.0f;
  this->print_images = false;
  this->equinox = 2000.0f;
};

IoFITS::IoFITS(std::string input, std::string output, std::string path)
    : Io(input, output, path) {
  this->M = 0;
  this->N = 0;
  this->normalization_factor = 1.0f;
  this->print_images = false;
  this->equinox = 2000.0f;
};

IoFITS::IoFITS(std::string input,
               std::string output,
               std::string path,
               int M,
               int N,
               float normalization_factor,
               bool print_images)
    : Io(input, output, path) {
  this->M = M;
  this->N = N;
  this->normalization_factor = normalization_factor;
  this->print_images = print_images;
  if (this->print_images && this->path != "")
    createFolder(this->path);
};

bool IoFITS::getPrintImages() {
  return this->print_images;
};

void IoFITS::setM(int M) {
  this->M = M;
};

void IoFITS::setN(int N) {
  this->N = N;
};

void IoFITS::setMN(int M, int N) {
  this->M = M;
  this->N = N;
};

void IoFITS::setEquinox(float equinox) {
  this->equinox = equinox;
};

void IoFITS::setFrame(std::string frame) {
  this->frame = frame;
};

void IoFITS::setRA(double ra) {
  this->ra = ra;
};

void IoFITS::setDec(double dec) {
  this->dec = dec;
};

void IoFITS::setRADec(double ra, double dec) {
  this->ra = ra;
  this->dec = dec;
};

void IoFITS::setNormalizationFactor(int normalization_factor) {
  this->normalization_factor = normalization_factor;
};

void IoFITS::setPrintImages(bool print_images) {
  this->print_images = print_images;
  if (this->print_images && this->path != "")
    createFolder(this->path);
};

gpuvmem::fits::FitsHeader IoFITS::readHeader(char* header_name) {
  this->input = std::string(header_name);
  try {
    return gpuvmem::fits::read_fits_header(this->input);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read header failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

gpuvmem::fits::FitsHeader IoFITS::readHeader(std::string header_name) {
  this->input = header_name;
  try {
    return gpuvmem::fits::read_fits_header(header_name);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read header failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

gpuvmem::fits::FitsHeader IoFITS::readHeader() {
  try {
    return gpuvmem::fits::read_fits_header(this->input);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read header failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<float> IoFITS::read_data_float_FITS() {
  try {
    return gpuvmem::fits::read_fits_image_float(this->input);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<float> IoFITS::read_data_float_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_image_float(filename ? filename : "");
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<float> IoFITS::read_data_float_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_image_float(filename);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS() {
  try {
    return gpuvmem::fits::read_fits_image_double(this->input);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_image_double(filename ? filename : "");
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_image_double(filename);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS() {
  try {
    return gpuvmem::fits::read_fits_image_int(this->input);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_image_int(filename ? filename : "");
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_image_int(filename);
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read image failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

void IoFITS::printImage(float* I,
                        char* path,
                        char* name_image,
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
  write_slice(I, path, name_image, units, iteration, index, fg_scale, M, N,
              ra_center, dec_center, frame, equinox, isInGPU, this->input);
}

void IoFITS::printImage(float* I,
                        char* name_image,
                        char* units,
                        int iteration,
                        int index,
                        bool isInGPU) {
  write_slice(I, getConstCharFromString(this->path), name_image, units,
              iteration, index, this->normalization_factor, this->M, this->N,
              this->ra, this->dec, this->frame, this->equinox, isInGPU,
              this->input);
}

void IoFITS::printImage(float* I,
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
  write_slice(I, getConstCharFromString(this->path),
              getConstCharFromString(this->output), units, iteration, index,
              fg_scale, M, N, ra_center, dec_center, frame, equinox, isInGPU,
              this->input);
}

void IoFITS::printImage(float* I,
                        char* name_image,
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
  write_slice(I, getConstCharFromString(this->path), name_image, units,
              iteration, index, fg_scale, M, N, ra_center, dec_center, frame,
              equinox, isInGPU, this->input);
}

void IoFITS::printNotPathImage(float* I,
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
  write_slice(I, "", getConstCharFromString(this->output), units, iteration,
              index, fg_scale, M, N, ra_center, dec_center, frame, equinox,
              isInGPU, this->input);
}

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
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
  write_slice(I, "", out_image, units, iteration, index, fg_scale, M, N,
              ra_center, dec_center, frame, equinox, isInGPU, this->input);
}

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
                               char* units,
                               int iteration,
                               int index,
                               bool isInGPU) {
  write_slice(I, "", out_image, units, iteration, index,
              this->normalization_factor, this->M, this->N, this->ra,
              this->dec, this->frame, this->equinox, isInGPU, this->input);
}

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
                               char* units,
                               int iteration,
                               int index,
                               float normalization_factor,
                               bool isInGPU) {
  write_slice(I, "", out_image, units, iteration, index, normalization_factor,
              this->M, this->N, this->ra, this->dec, this->frame, this->equinox,
              isInGPU, this->input);
}

void IoFITS::printNotPathImage(float* I,
                               char* units,
                               int iteration,
                               int index,
                               float normalization_factor,
                               bool isInGPU) {
  write_slice(I, "", getConstCharFromString(this->output), units, iteration,
              index, normalization_factor, this->M, this->N, this->ra,
              this->dec, this->frame, this->equinox, isInGPU, this->input);
}

void IoFITS::printNotNormalizedImage(float* I,
                                     char* name_image,
                                     char* units,
                                     int iteration,
                                     int index,
                                     bool isInGPU) {
  write_slice(I, getConstCharFromString(this->path), name_image, units,
              iteration, index, 1.0f, this->M, this->N, this->ra, this->dec,
              this->frame, this->equinox, isInGPU, this->input);
}

void IoFITS::printNormalizedImage(float* I,
                                  char* name_image,
                                  char* units,
                                  int iteration,
                                  int index,
                                  float scale,
                                  bool isInGPU) {
  write_slice(I, getConstCharFromString(this->path), name_image, units,
              iteration, index, scale, this->M, this->N, this->ra, this->dec,
              this->frame, this->equinox, isInGPU, this->input);
}

void IoFITS::printNotPathNotNormalizedImage(float* I,
                                            char* name_image,
                                            char* units,
                                            int iteration,
                                            int index,
                                            bool isInGPU) {
  write_slice(I, "", name_image, units, iteration, index, 1.0f, this->M,
              this->N, this->ra, this->dec, this->frame, this->equinox,
              isInGPU, this->input);
}

void IoFITS::printImageIteration(float* I,
                                 char const* name_image,
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
  size_t needed;
  char* full_name;

  needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
  full_name = (char*)malloc(needed * sizeof(char));
  snprintf(full_name, needed * sizeof(char), "%s_%d.fits", name_image,
           iteration);

  write_slice(I, getConstCharFromString(this->path), full_name, units,
              iteration, index, fg_scale, M, N, ra_center, dec_center, frame,
              equinox, isInGPU, this->input);
  free(full_name);
}

void IoFITS::printImageIteration(float* I,
                                 char const* name_image,
                                 char* units,
                                 int iteration,
                                 int index,
                                 bool isInGPU) {
  size_t needed;
  char* full_name;

  needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
  full_name = (char*)malloc(needed * sizeof(char));
  snprintf(full_name, needed * sizeof(char), "%s_%d.fits", name_image,
           iteration);

  write_slice(I, getConstCharFromString(this->path), full_name, units,
              iteration, index, this->normalization_factor, this->M, this->N,
              this->ra, this->dec, this->frame, this->equinox, isInGPU,
              this->input);
  free(full_name);
}

void IoFITS::printNotNormalizedImageIteration(float* I,
                                              char const* name_image,
                                              char* units,
                                              int iteration,
                                              int index,
                                              bool isInGPU) {
  size_t needed;
  char* full_name;

  needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
  full_name = (char*)malloc(needed * sizeof(char));
  snprintf(full_name, needed * sizeof(char), "%s_%d.fits", name_image,
           iteration);

  write_slice(I, getConstCharFromString(this->path), full_name, units,
              iteration, index, 1.0f, this->M, this->N, this->ra, this->dec,
              this->frame, this->equinox, isInGPU, this->input);
  free(full_name);
}

void IoFITS::printImageIteration(float* I,
                                 char* model_input,
                                 char* path,
                                 char const* name_image,
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
  size_t needed;
  char* full_name;

  needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
  full_name = (char*)malloc(needed * sizeof(char));
  snprintf(full_name, needed * sizeof(char), "%s_%d.fits", name_image,
           iteration);

  write_slice(I, path, full_name, units, iteration, index, fg_scale, M, N,
              ra_center, dec_center, frame, equinox, isInGPU,
              model_input ? model_input : this->input);
  free(full_name);
}

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               fitsfile* /*canvas*/,
                               char* out_image,
                               char* /*mempath*/,
                               int iteration,
                               float /*fg_scale*/,
                               long M,
                               long N,
                               int option,
                               bool isInGPU) {
  write_complex_slice(I, getConstCharFromString(this->input),
                      getConstCharFromString(this->path), out_image,
                      iteration, M, N, option, isInGPU);
};

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               fitsfile* /*canvas*/,
                               char* out_image,
                               char* /*mempath*/,
                               int iteration,
                               int option,
                               bool isInGPU) {
  write_complex_slice(I, getConstCharFromString(this->input),
                      getConstCharFromString(this->path), out_image,
                      iteration, this->M, this->N, option, isInGPU);
};

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               char* input,
                               char* path,
                               fitsfile* /*canvas*/,
                               char* out_image,
                               char* /*mempath*/,
                               int iteration,
                               float /*fg_scale*/,
                               long M,
                               long N,
                               int option,
                               bool isInGPU) {
  write_complex_slice(I, input, path, out_image, iteration, M, N, option, isInGPU);
};

void IoFITS::closeHeader(fitsfile* header) {
  // Legacy no-op: fitsfile* is no longer used with new FITS API
  // This method exists for interface compatibility only
  (void)header;
};

namespace {
Io* CreateIoFITS() {
  return new IoFITS;
}
const std::string IoFITSId = "IoFITS";
const bool RegisteredIoMS =
    registerCreationFunction<Io, std::string>(IoFITSId, CreateIoFITS);
};  // namespace
