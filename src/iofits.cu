#include "iofits.cuh"
#include "fits/fits_io.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

/** Convert gpuvmem::fits::FitsHeader to legacy headerValues for existing callers. */
headerValues to_header_values(const gpuvmem::fits::FitsHeader& h) {
  headerValues v;
  v.DELTAX = h.cdelt1;
  v.DELTAY = h.cdelt2;
  v.ra = h.crval1;
  v.dec = h.crval2;
  v.crpix1 = h.crpix1;
  v.crpix2 = h.crpix2;
  v.M = h.naxis1;
  v.N = h.naxis2;
  v.beam_bmaj = h.beam_maj;
  v.beam_bmin = h.beam_min;
  v.beam_bpa = h.beam_pa;
  v.beam_noise = h.noise_keyword;
  v.radesys = h.radesys;
  v.equinox = h.equinox;
  v.bitpix = h.bitpix;
  return v;
}

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
    fprintf(stderr, "FITS write failed: %s\n", e.what());
    std::exit(1);
  }
}

}  // namespace

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

headerValues IoFITS::readHeader(char* header_name) {
  this->input = std::string(header_name);
  try {
    return to_header_values(gpuvmem::fits::read_fits_header(this->input));
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read header failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

headerValues IoFITS::readHeader(std::string header_name) {
  this->input = header_name;
  try {
    return to_header_values(gpuvmem::fits::read_fits_header(header_name));
  } catch (const std::exception& e) {
    fprintf(stderr, "FITS read header failed: %s\n", e.what());
    std::exit(1);
  }
  return {};
}

headerValues IoFITS::readHeader() {
  try {
    return to_header_values(gpuvmem::fits::read_fits_header(this->input));
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
  std::vector<double> image;
  double* tmp;
  headerValues hdr =
      open_fits<double>(&tmp, getConstCharFromString(this->input), TDOUBLE);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<double> IoFITS::read_data_double_FITS(char* filename) {
  std::vector<double> image;
  double* tmp;
  headerValues hdr = open_fits<double>(&tmp, filename, TDOUBLE);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<double> IoFITS::read_data_double_FITS(std::string filename) {
  std::vector<double> image;
  double* tmp;
  headerValues hdr =
      open_fits<double>(&tmp, getConstCharFromString(filename), TDOUBLE);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<int> IoFITS::read_data_int_FITS() {
  std::vector<int> image;
  int* tmp;
  headerValues hdr =
      open_fits<int>(&tmp, getConstCharFromString(this->input), TINT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<int> IoFITS::read_data_int_FITS(char* filename) {
  std::vector<int> image;
  int* tmp;
  headerValues hdr = open_fits<int>(&tmp, filename, TINT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<int> IoFITS::read_data_int_FITS(std::string filename) {
  std::vector<int> image;
  int* tmp;
  headerValues hdr =
      open_fits<int>(&tmp, getConstCharFromString(filename), TINT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

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
                               fitsfile* canvas,
                               char* out_image,
                               char* mempath,
                               int iteration,
                               float fg_scale,
                               long M,
                               long N,
                               int option,
                               bool isInGPU) {
  OCopyFITSCufftComplex(I, getConstCharFromString(this->input),
                        getConstCharFromString(this->path), out_image,
                        iteration, fg_scale, M, N, option, isInGPU);
};

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               fitsfile* canvas,
                               char* out_image,
                               char* mempath,
                               int iteration,
                               int option,
                               bool isInGPU) {
  OCopyFITSCufftComplex(I, getConstCharFromString(this->input),
                        getConstCharFromString(this->path), out_image,
                        iteration, this->normalization_factor, this->M, this->N,
                        option, isInGPU);
};

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               char* input,
                               char* path,
                               fitsfile* canvas,
                               char* out_image,
                               char* mempath,
                               int iteration,
                               float fg_scale,
                               long M,
                               long N,
                               int option,
                               bool isInGPU) {
  OCopyFITSCufftComplex(I, input, path, out_image, iteration, fg_scale, M, N,
                        option, isInGPU);
};

void IoFITS::closeHeader(fitsfile* header) {
  closeFITS(header);
};

namespace {
Io* CreateIoFITS() {
  return new IoFITS;
}
const std::string IoFITSId = "IoFITS";
const bool RegisteredIoMS =
    registerCreationFunction<Io, std::string>(IoFITSId, CreateIoFITS);
};  // namespace
