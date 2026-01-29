#ifndef IO_CUH
#define IO_CUH

#include "MSFITSIO.cuh"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

typedef struct stat Stat;

class Io {
 public:
  Io() {
    this->input = "";
    this->output = "";
    this->path = "mem/";
  };

  Io(std::string input, std::string output, std::string path) {
    this->input = input;
    this->output = output;
    this->path = path;
    if (!checkLastTrailInPath()) {
      addCharToString(this->path, '/');
    }
  };

  virtual bool getPrintImages(){};
  virtual void setM(int M){};
  virtual void setN(int N){};
  virtual void setMN(int M, int N){};
  virtual void setEquinox(float equinox){};
  virtual void setFrame(std::string frame){};
  virtual void setRA(double ra){};
  virtual void setDec(double dec){};
  virtual void setRADec(double ra, double dec){};
  virtual void setNormalizationFactor(int normalization_factor){};
  virtual void setPrintImages(bool print_images){};

  virtual headerValues readHeader(){};
  virtual headerValues readHeader(char* header_name){};
  virtual headerValues readHeader(std::string header_name){};
  virtual std::vector<float> read_data_float_FITS(){};
  virtual std::vector<float> read_data_float_FITS(char* filename){};
  virtual std::vector<float> read_data_float_FITS(std::string filename){};
  virtual std::vector<double> read_data_double_FITS(){};
  virtual std::vector<double> read_data_double_FITS(char* filename){};
  virtual std::vector<double> read_data_double_FITS(std::string filename){};
  virtual std::vector<int> read_data_int_FITS(){};
  virtual std::vector<int> read_data_int_FITS(char* filename){};
  virtual std::vector<int> read_data_int_FITS(std::string filename){};
  virtual void printImage(float* I,
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
                          bool isInGPU){};
  virtual void printImage(float* I,
                          char* name_image,
                          char* units,
                          int iteration,
                          int index,
                          bool isInGPU){};
  virtual void printImage(float* I,
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
                          bool isInGPU){};
  virtual void printImage(float* I,
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
                          bool isInGPU){};
  virtual void printNotPathImage(float* I,
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
                                 bool isInGPU){};
  virtual void printNotPathImage(float* I,
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
                                 bool isInGPU){};
  virtual void printNotPathImage(float* I,
                                 char* out_image,
                                 char* units,
                                 int iteration,
                                 int index,
                                 bool isInGPU){};
  virtual void printNotPathImage(float* I,
                                 char* out_image,
                                 char* units,
                                 int iteration,
                                 int index,
                                 float normalization_factor,
                                 bool isInGPU){};
  virtual void printNotPathImage(float* I,
                                 char* units,
                                 int iteration,
                                 int index,
                                 float normalization_factor,
                                 bool isInGPU){};
  virtual void printNotNormalizedImage(float* I,
                                       char* name_image,
                                       char* units,
                                       int iteration,
                                       int index,
                                       bool isInGPU){};
  virtual void printNotPathNotNormalizedImage(float* I,
                                              char* name_image,
                                              char* units,
                                              int iteration,
                                              int index,
                                              bool isInGPU){};
  virtual void printImageIteration(float* I,
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
                                   bool isInGPU){};
  virtual void printImageIteration(float* I,
                                   char const* name_image,
                                   char* units,
                                   int iteration,
                                   int index,
                                   bool isInGPU){};
  virtual void printNotNormalizedImageIteration(float* I,
                                                char const* name_image,
                                                char* units,
                                                int iteration,
                                                int index,
                                                bool isInGPU){};
  virtual void printImageIteration(float* I,
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
                                   bool isInGPU){};
  virtual void printcuFFTComplex(cufftComplex* I,
                                 fitsfile* canvas,
                                 char* out_image,
                                 char* mempath,
                                 int iteration,
                                 float fg_scale,
                                 long M,
                                 long N,
                                 int option,
                                 bool isInGPU){};
  virtual void printcuFFTComplex(cufftComplex* I,
                                 fitsfile* canvas,
                                 char* out_image,
                                 char* mempath,
                                 int iteration,
                                 int option,
                                 bool isInGPU){};
  virtual void printcuFFTComplex(cufftComplex* I,
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
                                 bool isInGPU){};
  virtual void closeHeader(fitsfile* header){};

  virtual float getRandomProbability(){};
  virtual int getGridding(){};
  virtual bool getApplyNoiseInput(){};
  virtual bool getApplyNoiseOutput(){};
  virtual bool getWProjection(){};
  virtual bool getStoreModelVisInput(){};
  virtual std::string getDataColumnInput(){};
  virtual std::string getDataColumnOutput(){};
  virtual void setRandomProbability(float random_probability){};
  virtual void setGridding(int gridding){};
  virtual void setApplyNoiseInput(bool apply_noise_input){};
  virtual void setApplyNoiseOutput(bool apply_noise_output){};
  virtual void setNoise(bool input, bool output){};
  virtual void setWProjection(bool wprojection){};
  virtual void setStoreModelVisInput(bool store_model_vis_input){};
  virtual void setDataColumnInput(std::string datacolumn_input){};
  virtual void setDataColumnOutput(std::string datacolumn_output){};
  virtual void setDataColumns(std::string datacolumn_input,
                              std::string datacolumn_output){};

  virtual void read(std::vector<MSAntenna>& antennas,
                    std::vector<Field>& fields,
                    MSData* data){};
  virtual void read(char const* MS_name,
                    std::vector<MSAntenna>& antennas,
                    std::vector<Field>& fields,
                    MSData* data,
                    bool noise,
                    bool W_projection,
                    float random_probability,
                    int gridding){};
  virtual void read(char const* MS_name,
                    std::vector<MSAntenna>& antennas,
                    std::vector<Field>& fields,
                    MSData* data){};
  virtual void readSpecificColumn(std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data){};
  virtual void readSpecificColumn(std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data,
                                  std::string data_column){};
  virtual void readSpecificColumn(char const* MS_name,
                                  std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data,
                                  bool noise,
                                  bool W_projection,
                                  float random_probability,
                                  int gridding){};
  virtual void readSpecificColumn(char const* MS_name,
                                  std::string data_column,
                                  std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data,
                                  bool noise,
                                  bool W_projection,
                                  float random_probability,
                                  int gridding){};
  virtual void readSpecificColumn(char const* MS_name,
                                  std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data){};
  virtual void readSpecificColumn(char const* MS_name,
                                  std::string data_column,
                                  std::vector<MSAntenna>& antennas,
                                  std::vector<Field>& fields,
                                  MSData* data){};
  virtual void copy(char const* infile, char const* outfile){};
  virtual void copy(){};
  virtual void write(char const* outfile,
                     char const* out_col,
                     std::vector<Field>& fields,
                     MSData data,
                     float random_probability,
                     bool store_model_vis_input,
                     bool noise,
                     bool W_projection){};
  virtual void write(char const* out_col,
                     std::vector<Field>& fields,
                     MSData data){};
  virtual void write(char const* outfile,
                     char const* out_col,
                     std::vector<Field>& fields,
                     MSData data){};
  virtual void write(char const* outfile,
                     char const* out_col,
                     std::vector<Field>& fields,
                     MSData data,
                     bool store_model_vis_input){};
  virtual void write(char const* out_col,
                     std::vector<Field>& fields,
                     MSData data,
                     bool store_model){};
  virtual void writeSpecificColumn(char const* outfile,
                                   std::vector<Field>& fields,
                                   MSData data,
                                   float random_probability,
                                   bool store_model_vis_input,
                                   bool noise,
                                   bool W_projection){};
  virtual void writeSpecificColumn(std::vector<Field>& fields, MSData data){};
  virtual void writeSpecificColumn(char const* outfile,
                                   std::vector<Field>& fields,
                                   MSData data){};
  virtual void writeSpecificColumn(char const* outfile,
                                   std::vector<Field>& fields,
                                   MSData data,
                                   bool store_model_vis_input){};
  virtual void writeModelVisibilities(char const* outfile,
                                      std::vector<Field>& fields,
                                      MSData data){};
  virtual void writeModelVisibilities(std::vector<Field>& fields,
                                      MSData data){};
  virtual void writeResidualsAndModel(std::vector<Field>& fields,
                                      MSData data){};
  virtual void writeResidualsAndModel(std::string input,
                                      std::string output,
                                      std::vector<Field>& fields,
                                      MSData data){};

  void setPath(std::string pip) {
    this->path = pip;
    if (!checkLastTrailInPath()) {
      addCharToString(this->path, '/');
    }
  };

  void setPath(char* pip) {
    this->path = getStringFromChar(pip);
    if (!checkLastTrailInPath()) {
      addCharToString(this->path, '/');
    }
  };

  void setInput(std::string input) { this->input = input; };
  void setInput(char* input) { this->input = getStringFromChar(input); };

  void setOutput(std::string output) { this->output = output; };
  void setOutput(char* output) { this->output = getStringFromChar(output); };

 protected:
  std::string input;
  std::string output;
  std::string path;

  void addCharToString(std::string& input, char c) { input.push_back(c); };

  bool checkLastTrailInPath() {
    if (this->path.back() == '/')
      return true;
    else
      return false;
  };

  char* getCharFromString(std::string str) {
    char* writable = new char[str.size() + 1];
    std::copy(str.begin(), str.end(), writable);
    writable[str.size()] = '\0';
    return writable;
  };

  const char* getConstCharFromString(std::string str) { return str.c_str(); };

  std::string getStringFromChar(char* arr) { return std::string(arr); };

  bool createFolder(std::string str) {
    Stat st;
    bool status = false;
    if (stat(str.c_str(), &st) != 0) {
      /* Directory does not exist. EEXIST for race condition */
      if (mkdir(str.c_str(), 0700) != 0 && errno != EEXIST)
        status = true;
    }
    return status;
  };
};

#endif
