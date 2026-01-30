#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <cmath>
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

  // ---------------------------------------------------------------------------
  // Dimensions (legacy names: M = rows, N = columns)
  // ---------------------------------------------------------------------------
  int getImageCount() const { return image_count; };
  long getM() const { return M; };
  long getN() const { return N; };
  /** Number of columns (u direction); same as getN(). */
  long nx() const { return N; };
  /** Number of rows (v direction); same as getM(). */
  long ny() const { return M; };
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

  // ---------------------------------------------------------------------------
  // Image geometry (pixel scale and UV cell size); see docs/ms_dataset_migration_ref.md §13
  // ---------------------------------------------------------------------------
  /** Sky pixel size in RA (degrees), e.g. CDELT1. */
  double pixel_scale_ra_deg() const { return pixel_scale_ra_deg_; }
  /** Sky pixel size in Dec (degrees), e.g. CDELT2. */
  double pixel_scale_dec_deg() const { return pixel_scale_dec_deg_; }
  void set_pixel_scale_ra_deg(double v) { pixel_scale_ra_deg_ = v; }
  void set_pixel_scale_dec_deg(double v) { pixel_scale_dec_deg_ = v; }
  /** UV grid cell size in u (wavelengths). Computed: 1/(ny * pixel_scale_ra_rad). */
  double uv_cell_u() const {
    if (ny() <= 0 || pixel_scale_ra_deg_ == 0.0) return 0.0;
    double rad = pixel_scale_ra_deg_ * (3.14159265358979323846 / 180.0);
    return 1.0 / (static_cast<double>(ny()) * rad);
  }
  /** UV grid cell size in v (wavelengths). Computed: 1/(nx * pixel_scale_dec_rad). */
  double uv_cell_v() const {
    if (nx() <= 0 || pixel_scale_dec_deg_ == 0.0) return 0.0;
    double rad = pixel_scale_dec_deg_ * (3.14159265358979323846 / 180.0);
    return 1.0 / (static_cast<double>(nx()) * rad);
  }

  void setImageCount(int i) {
    this->image_count = i;
    // Resize minimal_pixel_values if needed
    if (minimal_pixel_values.size() != static_cast<size_t>(i)) {
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
    if (values.size() != static_cast<size_t>(image_count)) {
      this->image_count = values.size();
    }
  };
  void setMinimalPixelValue(int image_index, float value) {
    if (image_index >= 0 && image_index < static_cast<int>(minimal_pixel_values.size())) {
      minimal_pixel_values[image_index] = value;
    }
  };

 private:
  int image_count;
  long M;
  long N;
  double pixel_scale_ra_deg_{0.0};
  double pixel_scale_dec_deg_{0.0};
  float* image;
  float* error_image;
  imageMap* functionMapping;
  std::vector<float> minimal_pixel_values;  // Minimum pixel value for each image
};

#endif
