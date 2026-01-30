#ifndef GPUVMEM_FITS_FITS_IO_H
#define GPUVMEM_FITS_FITS_IO_H

#include <string>
#include <vector>

namespace gpuvmem {
namespace fits {

/** FITS image header (astrometry, beam, frame). Names follow FITS keywords where applicable. */
struct FitsHeader {
  long naxis1{0};        /**< NAXIS1: number of columns (x). */
  long naxis2{0};       /**< NAXIS2: number of rows (y). */
  double cdelt1{0};     /**< CDELT1: pixel scale in x (e.g. deg). */
  double cdelt2{0};     /**< CDELT2: pixel scale in y. */
  double crval1{0};      /**< CRVAL1: reference longitude/RA at reference pixel. */
  double crval2{0};     /**< CRVAL2: reference latitude/Dec at reference pixel. */
  double crpix1{0};     /**< CRPIX1: reference pixel (1-indexed) in x. */
  double crpix2{0};     /**< CRPIX2: reference pixel (1-indexed) in y. */
  double beam_maj{0};   /**< BMAJ: beam major axis (e.g. arcsec). */
  double beam_min{0};   /**< BMIN: beam minor axis. */
  double beam_pa{0};    /**< BPA: beam position angle (e.g. deg). */
  float noise_keyword{-1.0f}; /**< Optional NOISE header value. */
  std::string radesys;  /**< RADESYS: reference frame (e.g. "ICRS"). */
  float equinox{2000.0f}; /**< EQUINOX: equinox of coordinate system. */
  int bitpix{0};         /**< FITS bitpix (e.g. -32 for float). */
};

/** Options for writing a single 2D slice to a FITS file (template header + image). */
struct WriteFitsImageOptions {
  std::string output_path;      /**< Full path for output file (e.g. "mem/noise.fits"). */
  std::string header_template; /**< FITS file to copy header from (e.g. mod_in). */
  float* data{nullptr};         /**< Image buffer (host or device; see data_on_device). */
  long naxis1{0};               /**< NAXIS1: image width (columns). */
  long naxis2{0};               /**< NAXIS2: image height (rows). */
  int plane_index{0};           /**< Slice index: offset into buffer = naxis1*naxis2*plane_index. */
  const char* bunit{""};        /**< BUNIT: physical units (e.g. "JY/PIXEL"). */
  int niter{0};                 /**< NITER: iteration number (written to header). */
  float normalization_factor{1.0f}; /**< Scale factor applied to pixels if normalize is true. */
  double crval1{0};             /**< CRVAL1 for output header. */
  double crval2{0};             /**< CRVAL2 for output header. */
  std::string radesys{"ICRS"};  /**< RADESYS for output header. */
  float equinox{2000.0f};       /**< EQUINOX for output header. */
  bool normalize{true};        /**< If true, multiply pixels by normalization_factor before writing. */
  bool data_on_device{false};   /**< If true, data is on GPU; implementation copies to host. */
};

/** Read FITS image header from file. */
FitsHeader read_fits_header(const std::string& path);

/** Read full 2D image as float from first HDU. Returns row-major M*N floats. */
std::vector<float> read_fits_image_float(const std::string& path);

/** Write one 2D slice to a new FITS file: copy header from template, then write data (optionally normalized). */
void write_fits_image_slice(const WriteFitsImageOptions& opts);

}  // namespace fits
}  // namespace gpuvmem

#endif
