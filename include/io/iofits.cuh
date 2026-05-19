#ifndef IOFITS_CUH
#define IOFITS_CUH
#include "framework.cuh"

#include <optional>

class IoFITS : public Io {
 public:
  IoFITS();
  IoFITS(std::string input, std::string output, std::string path);
  IoFITS(std::string input,
         std::string output,
         std::string path,
         int M,
         int N,
         float normalization_factor,
         bool print_images);
  bool getPrintImages() override;
  void setM(int M) override;
  void setN(int N) override;
  void setEquinox(float equinox) override;
  void setFrame(std::string frame) override;
  void setRA(double ra) override;
  void setDec(double dec) override;
  void setRADec(double ra, double dec) override;
  void setMN(int M, int N) override;
  void setNormalizationFactor(int normalization_factor) override;
  void setPrintImages(bool print_images) override;
  void setModelFitsGeometry(std::optional<gpuvmem::ImagingHeader> geometry) override;
  gpuvmem::ImagingHeader readHeader(char* header_name) override;
  gpuvmem::ImagingHeader readHeader(std::string header_name) override;
  gpuvmem::ImagingHeader readHeader() override;
  std::vector<float> read_data_float_FITS() override;
  std::vector<float> read_data_float_FITS(char* filename) override;
  std::vector<float> read_data_float_FITS(std::string filename) override;
  std::vector<double> read_data_double_FITS() override;
  std::vector<double> read_data_double_FITS(char* filename) override;
  std::vector<double> read_data_double_FITS(std::string filename) override;
  std::vector<int> read_data_int_FITS() override;
  std::vector<int> read_data_int_FITS(char* filename) override;
  std::vector<int> read_data_int_FITS(std::string filename) override;

  /**
   * Read primary 2D float FITS (single HDU) into a new Image: GPU pixel buffer,
   * CDELT scales, and FITS metadata on the object. All file access is here or
   * in fits_io; Image never opens files. Caller must cudaFree(image->getImage())
   * and delete image.
   */
  static Image* readImageFromFits(const std::string& path, int cuda_device = 0);

  /**
   * Write one plane of a float Image to a new FITS file. Header from
   * image->imagingHeader(), setModelFitsGeometry(), header_template_path if set,
   * or this->input (same precedence as other FITS writes). Throws on failure.
   */
  void writeImageToFits(const Image* image, const std::string& output_path,
                        int plane_index = 0, const std::string& header_template_path = "",
                        bool data_on_device = true, const char* bunit = "JY/PIXEL",
                        int niter = 0, float normalization_factor = 1.0f, bool normalize = false);

  void printImage(float* I,
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
                  bool isInGPU) override;
  void printImage(float* I,
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
                  bool isInGPU) override;
  void printImage(float* I,
                  char* name_image,
                  char* units,
                  int iteration,
                  int index,
                  bool isInGPU) override;
  void printImage(float* I,
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
                  bool isInGPU) override;
  void printNotPathImage(float* I,
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
                         bool isInGPU) override;
  void printNotPathImage(float* I,
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
                         bool isInGPU) override;
  void printNotPathImage(float* I,
                         char* out_image,
                         char* units,
                         int iteration,
                         int index,
                         bool isInGPU) override;
  void printNotPathImage(float* I,
                         char* out_image,
                         char* units,
                         int iteration,
                         int index,
                         float normalization_factor,
                         bool isInGPU) override;
  void printNotPathImage(float* I,
                         char* units,
                         int iteration,
                         int index,
                         float normalization_factor,
                         bool isInGPU) override;
  void printNotNormalizedImage(float* I,
                               char* name_image,
                               char* units,
                               int iteration,
                               int index,
                               bool isInGPU) override;
  void printNormalizedImage(float* I,
                            char* name_image,
                            char* units,
                            int iteration,
                            int index,
                            float scale,
                            bool isInGPU) override;
  void printNotPathNotNormalizedImage(float* I,
                                      char* name_image,
                                      char* units,
                                      int iteration,
                                      int index,
                                      bool isInGPU) override;
  void printImageIteration(float* I,
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
                           bool isInGPU) override;
  void printImageIteration(float* I,
                           char const* name_image,
                           char* units,
                           int iteration,
                           int index,
                           bool isInGPU) override;
  void printNotNormalizedImageIteration(float* I,
                                        char const* name_image,
                                        char* units,
                                        int iteration,
                                        int index,
                                        bool isInGPU) override;
  void printImageIteration(float* I,
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
                           bool isInGPU) override;
  void printcuFFTComplex(cufftComplex* I,
                         fitsfile* canvas,
                         char* out_image,
                         char* mempath,
                         int iteration,
                         float fg_scale,
                         long M,
                         long N,
                         int option,
                         bool isInGPU) override;
  void printcuFFTComplex(cufftComplex* I,
                         fitsfile* canvas,
                         char* out_image,
                         char* mempath,
                         int iteration,
                         int option,
                         bool isInGPU) override;
  void printcuFFTComplex(cufftComplex* I,
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
                         bool isInGPU) override;
  void closeHeader(fitsfile* header) override;

  /** Empty string when using model_fits_geometry_ for FITS output headers. */
  std::string templatePathForWrites(const char* model_input_override) const;
  const gpuvmem::ImagingHeader* inlineHeaderForWrites() const;

 protected:
  int M;
  int N;
  double ra;   // in degrees
  double dec;  // in degrees
  std::string frame = "ICRS";
  float equinox = 2000.0;
  float normalization_factor;
  bool print_images;
  std::optional<gpuvmem::ImagingHeader> model_fits_geometry_;
};

#endif
