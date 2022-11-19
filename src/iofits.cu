#include "iofits.cuh"

IoFITS::IoFITS() : Io() {
  this->M = 0;
  this->N = 0;
  this->normalization_factor = 1.0f;
  this->print_images = false;
};

IoFITS::IoFITS(std::string input, std::string output, std::string path)
    : Io(input, output, path) {
  this->M = 0;
  this->N = 0;
  this->normalization_factor = 1.0f;
  this->print_images = false;
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
  return readFITSHeader(header_name);
};

headerValues IoFITS::readHeader(std::string header_name) {
  this->input = header_name;
  return readFITSHeader(getConstCharFromString(header_name));
};

headerValues IoFITS::readHeader() {
  return readFITSHeader(getConstCharFromString(this->input));
};

std::vector<float> IoFITS::read_data_float_FITS() {
  std::vector<float> image;
  float* tmp;
  headerValues hdr =
      open_fits<float>(&tmp, getConstCharFromString(this->input), TFLOAT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<float> IoFITS::read_data_float_FITS(char* filename) {
  std::vector<float> image;
  float* tmp;
  headerValues hdr = open_fits<float>(&tmp, filename, TFLOAT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

std::vector<float> IoFITS::read_data_float_FITS(std::string filename) {
  std::vector<float> image;
  float* tmp;
  headerValues hdr =
      open_fits<float>(&tmp, getConstCharFromString(filename), TFLOAT);
  int tmp_len = hdr.M * hdr.N;
  image.assign(tmp, tmp + tmp_len);
  free(tmp);
  return image;
};

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
  OCopyFITS(I, getConstCharFromString(this->input), path, name_image, units,
            iteration, index, fg_scale, M, N, ra_center, dec_center, frame,
            equinox, isInGPU);
};

void IoFITS::printImage(float* I,
                        char* name_image,
                        char* units,
                        int iteration,
                        int index,
                        bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), name_image, units, iteration,
            index, this->normalization_factor, this->M, this->N, this->ra,
            this->dec, this->frame, this->equinox, isInGPU);
};

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
  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path),
            getConstCharFromString(this->output), units, iteration, index,
            fg_scale, M, N, ra_center, dec_center, frame, equinox, isInGPU);
};

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
  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), name_image, units, iteration,
            index, fg_scale, M, N, ra_center, dec_center, frame, equinox,
            isInGPU);
};

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
  OCopyFITS(I, getConstCharFromString(this->input), "",
            getConstCharFromString(this->output), units, iteration, index,
            fg_scale, M, N, ra_center, dec_center, frame, equinox, isInGPU);
};

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
  OCopyFITS(I, getConstCharFromString(this->input), "", out_image, units,
            iteration, index, fg_scale, M, N, ra_center, dec_center, frame,
            equinox, isInGPU);
};

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
                               char* units,
                               int iteration,
                               int index,
                               bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input), "", out_image, units,
            iteration, index, this->normalization_factor, this->M, this->N,
            this->ra, this->dec, this->frame, this->equinox, isInGPU);
};

void IoFITS::printNotPathImage(float* I,
                               char* out_image,
                               char* units,
                               int iteration,
                               int index,
                               float normalization_factor,
                               bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input), "", out_image, units,
            iteration, index, normalization_factor, this->M, this->N, this->ra,
            this->dec, this->frame, this->equinox, isInGPU);
};

void IoFITS::printNotPathImage(float* I,
                               char* units,
                               int iteration,
                               int index,
                               float normalization_factor,
                               bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input), "",
            getConstCharFromString(this->output), units, iteration, index,
            normalization_factor, this->M, this->N, this->ra, this->dec,
            this->frame, this->equinox, isInGPU);
};

void IoFITS::printNotNormalizedImage(float* I,
                                     char* name_image,
                                     char* units,
                                     int iteration,
                                     int index,
                                     bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), name_image, units, iteration,
            index, 1.0f, this->M, this->N, this->ra, this->dec, this->frame,
            this->equinox, isInGPU);
};

void IoFITS::printNotPathNotNormalizedImage(float* I,
                                            char* name_image,
                                            char* units,
                                            int iteration,
                                            int index,
                                            bool isInGPU) {
  OCopyFITS(I, getConstCharFromString(this->input), "", name_image, units,
            iteration, index, 1.0f, this->M, this->N, this->ra, this->dec,
            this->frame, this->equinox, isInGPU);
};

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

  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), full_name, units, iteration,
            index, fg_scale, M, N, ra_center, dec_center, frame, equinox,
            isInGPU);
  free(full_name);
};

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

  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), full_name, units, iteration,
            index, this->normalization_factor, this->M, this->N, this->ra,
            this->dec, this->frame, this->equinox, isInGPU);
  free(full_name);
};

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

  OCopyFITS(I, getConstCharFromString(this->input),
            getConstCharFromString(this->path), full_name, units, iteration,
            index, 1.0f, this->M, this->N, this->ra, this->dec, this->frame,
            this->equinox, isInGPU);
  free(full_name);
};

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

  OCopyFITS(I, model_input, path, full_name, units, iteration, index, fg_scale,
            M, N, ra_center, dec_center, frame, equinox, isInGPU);
  free(full_name);
};

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
