#ifndef GPUVMEM_CLASSES_IMAGING_HEADER_HH
#define GPUVMEM_CLASSES_IMAGING_HEADER_HH

#include "fits/fits_io.h"
#include <string>

namespace gpuvmem {

/**
 * In-memory image astrometry and metadata (imager-native: 0-based reference pixel).
 * FITS files use 1-based CRPIX; convert on read with imaging_header_from_fits_wire()
 * and on write with fits_wire_header_from_imaging().
 */
struct ImagingHeader {
  long width_columns{0};   /**< NAXIS1 (fast axis), same as Image::getN(). */
  long height_rows{0};   /**< NAXIS2, same as Image::getM(). */
  double reference_column{0.0};
  double reference_row{0.0};
  double cdelt1{0.0};
  double cdelt2{0.0};
  double crval1{0.0};
  double crval2{0.0};
  double beam_maj{0.0};
  double beam_min{0.0};
  double beam_pa{0.0};
  float noise_keyword{-1.0f};
  std::string radesys;
  float equinox{2000.0f};
  int bitpix{0};
};

/** Build internal header from FITS primary HDU keywords (CRPIX 1-based → reference_* 0-based). */
inline ImagingHeader imaging_header_from_fits_wire(const fits::FitsHeader& w) {
  ImagingHeader h;
  h.width_columns = w.naxis1;
  h.height_rows = w.naxis2;
  h.cdelt1 = w.cdelt1;
  h.cdelt2 = w.cdelt2;
  h.crval1 = w.crval1;
  h.crval2 = w.crval2;
  h.beam_maj = w.beam_maj;
  h.beam_min = w.beam_min;
  h.beam_pa = w.beam_pa;
  h.noise_keyword = w.noise_keyword;
  h.radesys = w.radesys;
  h.equinox = w.equinox;
  h.bitpix = w.bitpix;
  if (w.crpix1 > 0.0 && w.crpix2 > 0.0) {
    h.reference_column = w.crpix1 - 1.0;
    h.reference_row = w.crpix2 - 1.0;
  } else if (w.naxis1 > 0 && w.naxis2 > 0) {
    h.reference_column = static_cast<double>(w.naxis1 / 2);
    h.reference_row = static_cast<double>(w.naxis2 / 2);
  }
  return h;
}

/** FITS wire header for CFITSIO write; naxis* must match image plane size. */
inline fits::FitsHeader fits_wire_header_from_imaging(const ImagingHeader& ih, long naxis1,
                                                     long naxis2) {
  fits::FitsHeader w;
  w.naxis1 = naxis1;
  w.naxis2 = naxis2;
  w.cdelt1 = ih.cdelt1;
  w.cdelt2 = ih.cdelt2;
  w.crval1 = ih.crval1;
  w.crval2 = ih.crval2;
  w.crpix1 = ih.reference_column + 1.0;
  w.crpix2 = ih.reference_row + 1.0;
  w.beam_maj = ih.beam_maj;
  w.beam_min = ih.beam_min;
  w.beam_pa = ih.beam_pa;
  w.noise_keyword = ih.noise_keyword;
  w.radesys = ih.radesys;
  w.equinox = ih.equinox;
  w.bitpix = ih.bitpix;
  return w;
}

}  // namespace gpuvmem

#endif
