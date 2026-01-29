#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <vector>

typedef struct functionMap {
  void (*newP)(float*, float*, float, int);
  void (*evaluateXt)(float*, float*, float*, float, int);  // Fixed: added missing float parameter
} imageMap;

class Image {
 public:
  Image(float* image, int image_count, long M = 0, long N = 0) {
    this->image = image;
    this->image_count = image_count;
    this->M = M;
    this->N = N;
    // Initialize minimal_pixel_values with default [0.0] for each image
    this->minimal_pixel_values = std::vector<float>(image_count, 0.0f);
  };
  
  int getImageCount() const { return image_count; };
  long getM() const { return M; };
  long getN() const { return N; };
  float* getImage() { return image; };
  float* getErrorImage() { return error_image; };
  imageMap* getFunctionMapping() { return functionMapping; };
  float getMinimalPixelValue(int image_index) const {
    if (image_index >= 0 && image_index < minimal_pixel_values.size()) {
      return minimal_pixel_values[image_index];
    }
    return 0.0f;  // Default fallback
  };
  const std::vector<float>& getMinimalPixelValues() const {
    return minimal_pixel_values;
  };
  
  void setImageCount(int i) { 
    this->image_count = i;
    // Resize minimal_pixel_values if needed
    if (minimal_pixel_values.size() != i) {
      minimal_pixel_values.resize(i, 0.0f);
    }
  };
  void setM(long M) { this->M = M; };
  void setN(long N) { this->N = N; };
  void setMN(long M, long N) {
    this->M = M;
    this->N = N;
  };
  void setErrorImage(float* f) { this->error_image = f; };
  void setImage(float* i) { this->image = i; };
  void setFunctionMapping(imageMap* f) { this->functionMapping = f; };
  void setMinimalPixelValues(const std::vector<float>& values) {
    this->minimal_pixel_values = values;
    // Update image_count if values size differs
    if (values.size() != image_count) {
      this->image_count = values.size();
    }
  };
  void setMinimalPixelValue(int image_index, float value) {
    if (image_index >= 0 && image_index < minimal_pixel_values.size()) {
      minimal_pixel_values[image_index] = value;
    }
  };

 private:
  int image_count;
  long M;
  long N;
  float* image;
  float* error_image;
  imageMap* functionMapping;
  std::vector<float> minimal_pixel_values;  // Minimum pixel value for each image
};

#endif
