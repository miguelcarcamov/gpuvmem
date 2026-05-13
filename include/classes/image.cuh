#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "classes/imaging_header.hh"
#include <cmath>
#include <vector>

typedef struct functionMap {
  void (*newP)(float*, float*, float, int);
  void (*evaluateXt)(float*, float*, float*, float, int);  // Fixed: added missing float parameter
} imageMap;

/** Host imaging grid geometry for synthesis (pixel scales, UV cell, reference pixel in 0-based indices). */
struct ImagingGeometry {
  double deltau{0.0};
  double deltav{0.0};
  double delta_x_deg{0.0};  /**< Legacy DELTAX: deg/pixel column (e.g. CDELT1). */
  double delta_y_deg{0.0};  /**< Legacy DELTAY (e.g. CDELT2). */
  /** 0-based column index j of the WCS reference pixel (width N / NAXIS1). On FITS write: CRPIX1 = this + 1. */
  double reference_column{0.0};
  /** 0-based row index k of the WCS reference pixel (height M / NAXIS2). On FITS write: CRPIX2 = this + 1. */
  double reference_row{0.0};
};

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
  float* getImage() const { return image; };
  float* getErrorImage() { return error_image; };
  imageMap* getFunctionMapping() { return functionMapping; };
  float getMinimalPixelValue(int image_index) const {
    if (image_index >= 0 && image_index < minimal_pixel_values.size()) {
      return minimal_pixel_values[image_index];
    }
    return 0.0f;  // Default value
  };
  const std::vector<float>& getMinimalPixelValues() const {
    return minimal_pixel_values;
  };

  // ---------------------------------------------------------------------------
  // Image geometry (pixel scale and UV cell size); see docs/ms_dataset_migration_ref.md §13
  // ---------------------------------------------------------------------------
  /** Sky pixel size in RA (degrees), e.g. CDELT1; same as imaging_header().cdelt1 when a header is set. */
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
  /** UV grid cell in v (λ); matches legacy deltav = 1/(nx * |CDELT2| in rad). */
  double uv_cell_v() const {
    if (nx() <= 0 || pixel_scale_dec_deg_ == 0.0) return 0.0;
    double rad = pixel_scale_dec_deg_ * (3.14159265358979323846 / 180.0);
    return 1.0 / (static_cast<double>(nx()) * rad);
  }

  /** Geometry for χ² / measurement operator (from stored scales + optional imaging header). */
  ImagingGeometry imaging_geometry() const {
    ImagingGeometry g;
    g.delta_x_deg = pixel_scale_ra_deg_;
    g.delta_y_deg = pixel_scale_dec_deg_;
    g.deltau = uv_cell_u();
    g.deltav = uv_cell_v();
    if (has_imaging_header_) {
      g.reference_column = imaging_header_.reference_column;
      g.reference_row = imaging_header_.reference_row;
    } else {
      g.reference_column = static_cast<double>(N / 2);
      g.reference_row = static_cast<double>(M / 2);
    }
    return g;
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

  /** In-memory astrometry (0-based reference pixel). Set after reading FITS or building a synthetic grid. */
  void setImagingHeader(const gpuvmem::ImagingHeader& h) {
    imaging_header_ = h;
    has_imaging_header_ = true;
    pixel_scale_ra_deg_ = h.cdelt1;
    pixel_scale_dec_deg_ = h.cdelt2;
  }
  void clearImagingHeader() {
    has_imaging_header_ = false;
    imaging_header_ = {};
  }
  bool hasImagingHeader() const { return has_imaging_header_; }
  const gpuvmem::ImagingHeader& imagingHeader() const { return imaging_header_; }

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
  gpuvmem::ImagingHeader imaging_header_{};
  bool has_imaging_header_{false};
};

#endif
