#include "io/iofits.cuh"
#include "classes/imaging_header.hh"
#include "fits/fits_io.h"
#include <helper_cuda.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

/** Build WriteFitsImageOptions from IoFITS state and write one slice; exits on error. */
void write_slice(float* I, const char* path, const char* name_image,
                 const char* units, int iteration, int index, float fg_scale,
                 long M, long N, double ra_center, double dec_center,
                 std::string frame, float equinox, bool isInGPU,
                 const std::string& template_path,
                 const gpuvmem::ImagingHeader* inline_primary_header) {
  gpuvmem::fits::WriteFitsImageOptions opts;
  opts.header_template = template_path;
  opts.inline_primary_header = inline_primary_header;
  opts.output_path = (path && path[0]) ? (std::string(path) + name_image) : name_image;
  opts.data = I;
  opts.naxis1 = N;
  opts.naxis2 = M;
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
    std::cerr
        << "FITS output failed (model FITS header or in-memory WCS template; "
           "see message for read vs write): "
        << e.what() << '\n';
    std::exit(1);
  }
}

/** Build WriteFitsComplexImageOptions and write complex image; exits on error. */
void write_complex_slice(cufftComplex* I, const std::string& template_path,
                         const gpuvmem::ImagingHeader* inline_primary_header,
                         const char* path, const char* out_image, int iteration,
                         long M, long N, int option, bool isInGPU) {
  gpuvmem::fits::WriteFitsComplexImageOptions opts;
  opts.header_template = template_path;
  opts.inline_primary_header = inline_primary_header;
  opts.data = I;
  opts.naxis1 = N;
  opts.naxis2 = M;
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
      std::cerr << "Invalid case to FITS\n";
      std::exit(-1);
      break;
    default:
      std::cerr << "Invalid case to FITS\n";
      std::exit(-1);
      break;
  }
  opts.output_path = output_path;
  
  try {
    gpuvmem::fits::write_fits_image_complex(opts);
  } catch (const std::exception& e) {
    std::cerr << "FITS complex write failed: " << e.what() << '\n';
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

void IoFITS::setModelFitsGeometry(std::optional<gpuvmem::ImagingHeader> geometry) {
  model_fits_geometry_ = std::move(geometry);
}

std::string IoFITS::templatePathForWrites(const char* model_input_override) const {
  if (model_fits_geometry_.has_value()) return {};
  if (model_input_override && model_input_override[0])
    return std::string(model_input_override);
  return this->input;
}

const gpuvmem::ImagingHeader* IoFITS::inlineHeaderForWrites() const {
  return model_fits_geometry_.has_value() ? &*model_fits_geometry_ : nullptr;
}

gpuvmem::ImagingHeader IoFITS::readHeader(char* header_name) {
  return readHeader(std::string(header_name ? header_name : ""));
}

gpuvmem::ImagingHeader IoFITS::readHeader(std::string header_name) {
  if (!header_name.empty())
    this->input = std::move(header_name);
  if (model_fits_geometry_.has_value() && this->input.empty())
    return *model_fits_geometry_;
  if (this->input.empty()) {
    std::cerr
        << "FITS read header failed: empty model path and no in-memory geometry\n";
    std::exit(1);
  }
  try {
    return gpuvmem::imaging_header_from_fits_wire(gpuvmem::fits::read_fits_header(this->input));
  } catch (const std::exception& e) {
    std::cerr << "FITS read header failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

gpuvmem::ImagingHeader IoFITS::readHeader() {
  return readHeader(std::string());
}

std::vector<float> IoFITS::read_data_float_FITS() {
  try {
    return gpuvmem::fits::read_fits_float_image(this->input).pixels;
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<float> IoFITS::read_data_float_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_float_image(filename ? filename : "").pixels;
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<float> IoFITS::read_data_float_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_float_image(filename).pixels;
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS() {
  try {
    return gpuvmem::fits::read_fits_image_double(this->input);
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_image_double(filename ? filename : "");
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<double> IoFITS::read_data_double_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_image_double(filename);
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS() {
  try {
    return gpuvmem::fits::read_fits_image_int(this->input);
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS(char* filename) {
  try {
    return gpuvmem::fits::read_fits_image_int(filename ? filename : "");
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
    std::exit(1);
  }
  return {};
}

std::vector<int> IoFITS::read_data_int_FITS(std::string filename) {
  try {
    return gpuvmem::fits::read_fits_image_int(filename);
  } catch (const std::exception& e) {
    std::cerr << "FITS read image failed: " << e.what() << '\n';
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
              ra_center, dec_center, frame, equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              ra_center, dec_center, frame, equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
                               char* units,
                               int iteration,
                               int index,
                               bool isInGPU) {
  write_slice(I, "", out_image, units, iteration, index,
              this->normalization_factor, this->M, this->N, this->ra,
              this->dec, this->frame, this->equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printNotPathImage(float* I,
                               char* units,
                               int iteration,
                               int index,
                               float normalization_factor,
                               bool isInGPU) {
  write_slice(I, "", getConstCharFromString(this->output), units, iteration,
              index, normalization_factor, this->M, this->N, this->ra,
              this->dec, this->frame, this->equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printNotNormalizedImage(float* I,
                                     char* name_image,
                                     char* units,
                                     int iteration,
                                     int index,
                                     bool isInGPU) {
  write_slice(I, getConstCharFromString(this->path), name_image, units,
              iteration, index, 1.0f, this->M, this->N, this->ra, this->dec,
              this->frame, this->equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
              this->frame, this->equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printNotPathNotNormalizedImage(float* I,
                                            char* name_image,
                                            char* units,
                                            int iteration,
                                            int index,
                                            bool isInGPU) {
  write_slice(I, "", name_image, units, iteration, index, 1.0f, this->M,
              this->N, this->ra, this->dec, this->frame, this->equinox,
              isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
  const std::string full_name =
      std::string(name_image) + "_" + std::to_string(iteration) + ".fits";

  write_slice(I, getConstCharFromString(this->path), full_name.c_str(), units,
              iteration, index, fg_scale, M, N, ra_center, dec_center, frame,
              equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printImageIteration(float* I,
                                 char const* name_image,
                                 char* units,
                                 int iteration,
                                 int index,
                                 bool isInGPU) {
  const std::string full_name =
      std::string(name_image) + "_" + std::to_string(iteration) + ".fits";

  write_slice(I, getConstCharFromString(this->path), full_name.c_str(), units,
              iteration, index, this->normalization_factor, this->M, this->N,
              this->ra, this->dec, this->frame, this->equinox, isInGPU,
              templatePathForWrites(nullptr), inlineHeaderForWrites());
}

void IoFITS::printNotNormalizedImageIteration(float* I,
                                              char const* name_image,
                                              char* units,
                                              int iteration,
                                              int index,
                                              bool isInGPU) {
  const std::string full_name =
      std::string(name_image) + "_" + std::to_string(iteration) + ".fits";

  write_slice(I, getConstCharFromString(this->path), full_name.c_str(), units,
              iteration, index, 1.0f, this->M, this->N, this->ra, this->dec,
              this->frame, this->equinox, isInGPU, templatePathForWrites(nullptr), inlineHeaderForWrites());
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
  const std::string full_name =
      std::string(name_image) + "_" + std::to_string(iteration) + ".fits";

  write_slice(I, path, full_name.c_str(), units, iteration, index, fg_scale, M, N,
              ra_center, dec_center, frame, equinox, isInGPU,
              templatePathForWrites(model_input), inlineHeaderForWrites());
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
  write_complex_slice(I, templatePathForWrites(nullptr), inlineHeaderForWrites(),
                      getConstCharFromString(this->path), out_image, iteration, M, N,
                      option, isInGPU);
};

void IoFITS::printcuFFTComplex(cufftComplex* I,
                               fitsfile* /*canvas*/,
                               char* out_image,
                               char* /*mempath*/,
                               int iteration,
                               int option,
                               bool isInGPU) {
  write_complex_slice(I, templatePathForWrites(nullptr), inlineHeaderForWrites(),
                      getConstCharFromString(this->path), out_image, iteration,
                      this->M, this->N, option, isInGPU);
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
  const std::string tpl =
      (input && input[0]) ? std::string(input) : templatePathForWrites(nullptr);
  const gpuvmem::ImagingHeader* inl =
      (input && input[0]) ? nullptr : inlineHeaderForWrites();
  write_complex_slice(I, tpl, inl, path, out_image, iteration, M, N, option,
                      isInGPU);
};

Image* IoFITS::readImageFromFits(const std::string& path, int cuda_device) {
  if (path.empty()) throw std::runtime_error("readImageFromFits: empty path");
  gpuvmem::fits::FitsFloatImage data = gpuvmem::fits::read_fits_float_image(path);
  const long n1 = data.header.naxis1;
  const long n2 = data.header.naxis2;
  const size_t count = static_cast<size_t>(n1) * static_cast<size_t>(n2);
  if (data.pixels.size() != count)
    throw std::runtime_error("readImageFromFits: pixel count does not match NAXIS1*NAXIS2");
  float* d = nullptr;
  checkCudaErrors(cudaSetDevice(cuda_device));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d), count * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d, data.pixels.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  Image* img = new Image(d, 1, n2, n1);
  img->setImagingHeader(gpuvmem::imaging_header_from_fits_wire(data.header));
  return img;
}

void IoFITS::writeImageToFits(const Image* image, const std::string& output_path, int plane_index,
                              const std::string& header_template_path, bool data_on_device,
                              const char* bunit, int niter, float normalization_factor,
                              bool normalize) {
  if (!image) throw std::runtime_error("writeImageToFits: null Image");
  if (plane_index < 0 || plane_index >= image->getImageCount())
    throw std::runtime_error("writeImageToFits: plane_index out of range");

  gpuvmem::fits::WriteFitsImageOptions opts;
  opts.output_path = output_path;
  opts.data = image->getImage();
  opts.naxis1 = image->getN();
  opts.naxis2 = image->getM();
  opts.plane_index = plane_index;
  opts.bunit = bunit ? bunit : "";
  opts.niter = niter;
  opts.normalization_factor = normalization_factor;
  opts.normalize = normalize;
  opts.data_on_device = data_on_device;

  const gpuvmem::ImagingHeader* inline_hdr = nullptr;
  if (image->hasImagingHeader())
    inline_hdr = &image->imagingHeader();
  else
    inline_hdr = inlineHeaderForWrites();

  if (inline_hdr) {
    opts.inline_primary_header = inline_hdr;
    opts.header_template.clear();
  } else {
    opts.inline_primary_header = nullptr;
    opts.header_template =
        !header_template_path.empty() ? header_template_path : templatePathForWrites(nullptr);
  }

  if (image->hasImagingHeader()) {
    const gpuvmem::ImagingHeader& h = image->imagingHeader();
    opts.crval1 = h.crval1;
    opts.crval2 = h.crval2;
    opts.radesys = h.radesys.empty() ? frame : h.radesys;
    opts.equinox = h.equinox;
  } else {
    opts.crval1 = ra;
    opts.crval2 = dec;
    opts.radesys = frame;
    opts.equinox = equinox;
  }

  try {
    gpuvmem::fits::write_fits_image_slice(opts);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("writeImageToFits: ") + e.what());
  }
}

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
