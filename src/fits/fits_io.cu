/* FITS I/O in gpuvmem: CCfits (mandatory). RAII, options struct, multi-plane.
 * See docs/fits_io_refactor.md. CCfits wraps CFITSIO; we link both.
 */

#include "fits/fits_io.h"
#include "classes/imaging_header.hh"
#include <CCfits/FITS.h>
#include <CCfits/PHDU.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fitsio.h>
#include <helper_cuda.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpuvmem {
namespace fits {

static void throw_from_ccfits(const CCfits::FitsException& e) {
  throw std::runtime_error(std::string("CCfits: ") + e.message());
}

static void throw_cfitsio(int status, const char* context) {
  if (!status) return;
  char err_text[120];
  fits_get_errstatus(status, err_text);
  throw std::runtime_error(std::string(context) + ": " + err_text);
}

/** Close a fitsfile* with CFITSIO (used by unique_ptr deleter). */
struct CloseFitsFile {
  void operator()(fitsfile* fp) const {
    if (!fp) return;
    int st = 0;
    fits_close_file(fp, &st);
    (void)st;
  }
};

static void apply_output_image_keys(fitsfile* outf, long naxis1, long naxis2, const char* bunit,
                             int niter, double crval1, double crval2,
                             const std::string& radesys_in, float equinox,
                             bool update_pointing_keywords, int& status) {
  char bunit_buf[72] = "";
  if (bunit) snprintf(bunit_buf, sizeof(bunit_buf), "%s", bunit);
  fits_update_key(outf, TSTRING, "BUNIT", bunit_buf, "Unit of measurement", &status);
  if (status) status = 0;
  int niter_val = niter;
  fits_update_key(outf, TINT, "NITER", &niter_val,
                  "Number of iteration in gpuvmem software", &status);
  if (status) status = 0;
  int na1 = static_cast<int>(naxis1), na2 = static_cast<int>(naxis2);
  fits_update_key(outf, TINT, "NAXIS1", &na1, "", &status);
  if (status) status = 0;
  fits_update_key(outf, TINT, "NAXIS2", &na2, "", &status);
  if (status) status = 0;
  if (update_pointing_keywords) {
    std::string radesys_str = radesys_in;
    if (radesys_str.empty()) radesys_str = "ICRS";
    char radesys_buf[72];
    snprintf(radesys_buf, sizeof(radesys_buf), "%.70s", radesys_str.c_str());
    fits_update_key(outf, TSTRING, "RADESYS", radesys_buf, "Changed by gpuvmem",
                    &status);
    if (status) status = 0;
    float equinox_val = equinox;
    fits_update_key(outf, TFLOAT, "EQUINOX", &equinox_val, "Changed by gpuvmem",
                    &status);
    if (status) status = 0;
    double crval1_val = crval1, crval2_val = crval2;
    fits_update_key(outf, TDOUBLE, "CRVAL1", &crval1_val, "Changed by gpuvmem",
                    &status);
    if (status) status = 0;
    fits_update_key(outf, TDOUBLE, "CRVAL2", &crval2_val, "Changed by gpuvmem",
                    &status);
    if (status) status = 0;
  }
}

static void write_primary_wcs_from_header(fitsfile* fp, const FitsHeader& hdr, int& status) {
  double crval1 = hdr.crval1;
  double crval2 = hdr.crval2;
  double crpix1 = hdr.crpix1;
  double crpix2 = hdr.crpix2;
  double cdelt1 = hdr.cdelt1;
  double cdelt2 = hdr.cdelt2;
  fits_update_key(fp, TDOUBLE, "CRVAL1", &crval1, "deg", &status);
  if (status) status = 0;
  fits_update_key(fp, TDOUBLE, "CRVAL2", &crval2, "deg", &status);
  if (status) status = 0;
  fits_update_key(fp, TDOUBLE, "CRPIX1", &crpix1, "ref pixel (1-based)", &status);
  if (status) status = 0;
  fits_update_key(fp, TDOUBLE, "CRPIX2", &crpix2, "ref pixel (1-based)", &status);
  if (status) status = 0;
  fits_update_key(fp, TDOUBLE, "CDELT1", &cdelt1, "deg/pixel", &status);
  if (status) status = 0;
  fits_update_key(fp, TDOUBLE, "CDELT2", &cdelt2, "deg/pixel", &status);
  if (status) status = 0;
  char ctype_ra[] = "RA---SIN";
  char ctype_dec[] = "DEC--SIN";
  fits_update_key(fp, TSTRING, "CTYPE1", ctype_ra, "", &status);
  if (status) status = 0;
  fits_update_key(fp, TSTRING, "CTYPE2", ctype_dec, "", &status);
  if (status) status = 0;
  char cunit1[] = "deg";
  char cunit2[] = "deg";
  fits_update_key(fp, TSTRING, "CUNIT1", cunit1, "", &status);
  if (status) status = 0;
  fits_update_key(fp, TSTRING, "CUNIT2", cunit2, "", &status);
  if (status) status = 0;
  std::string rsys = hdr.radesys.empty() ? std::string("ICRS") : hdr.radesys;
  char radesys_buf[72];
  snprintf(radesys_buf, sizeof(radesys_buf), "%.70s", rsys.c_str());
  fits_update_key(fp, TSTRING, "RADESYS", radesys_buf, "", &status);
  if (status) status = 0;
  float equinox_val = hdr.equinox;
  fits_update_key(fp, TFLOAT, "EQUINOX", &equinox_val, "", &status);
  if (status) status = 0;
  if (hdr.beam_maj > 0.0) {
    double bmaj = hdr.beam_maj, bmin = hdr.beam_min, bpa = hdr.beam_pa;
    fits_update_key(fp, TDOUBLE, "BMAJ", &bmaj, "", &status);
    if (status) status = 0;
    fits_update_key(fp, TDOUBLE, "BMIN", &bmin, "", &status);
    if (status) status = 0;
    fits_update_key(fp, TDOUBLE, "BPA", &bpa, "", &status);
    if (status) status = 0;
  }
}

static void write_float_image_from_inline_header_cfitsio(
    const FitsHeader& hdr,
    const std::string& output_path_raw,
    long naxis1,
    long naxis2,
    long elements,
    float* write_ptr,
    const char* bunit,
    int niter,
    double crval1,
    double crval2,
    const std::string& radesys_in,
    float equinox,
    bool update_pointing_keywords) {
  if (hdr.naxis1 != naxis1 || hdr.naxis2 != naxis2)
    throw std::runtime_error(
        "write_float_image_from_inline_header_cfitsio: header NAXIS mismatch with write size");
  int status = 0;
  std::string outp = output_path_raw;
  if (outp.empty() || outp[0] != '!') outp = "!" + outp;
  std::vector<char> outp_path(outp.begin(), outp.end());
  outp_path.push_back('\0');
  fitsfile* outf = nullptr;
  fits_create_file(&outf, outp_path.data(), &status);
  throw_cfitsio(status, "inline header: fits_create_file");
  std::unique_ptr<fitsfile, CloseFitsFile> close_out(outf);
  long naxes[2] = {naxis1, naxis2};
  fits_create_img(outf, FLOAT_IMG, 2, naxes, &status);
  throw_cfitsio(status, "inline header: fits_create_img");
  write_primary_wcs_from_header(outf, hdr, status);
  apply_output_image_keys(outf, naxis1, naxis2, bunit, niter, crval1, crval2,
                          radesys_in, equinox, update_pointing_keywords, status);
  fits_write_img(outf, TFLOAT, 1, elements, write_ptr, &status);
  throw_cfitsio(status, "inline header: fits_write_img");
}

/**
 * Write a 2D float image using CFITSIO only: open template primary HDU,
 * copy_header to a new file, update keys, write pixels.
 * Avoids CCfits' full-file parse, which fails on some CASA-style model FITS
 * files that set EXTEND=T but have no valid extension HDU after the primary.
 */
static void write_float_image_from_template_cfitsio(
    const std::string& template_path,
    const std::string& output_path_raw,
    long naxis1,
    long naxis2,
    long elements,
    float* write_ptr,
    const char* bunit,
    int niter,
    double crval1,
    double crval2,
    const std::string& radesys_in,
    float equinox,
    bool update_pointing_keywords) {
  int status = 0;
  fitsfile* tin = nullptr;
  fits_open_file(&tin, template_path.c_str(), READONLY, &status);
  throw_cfitsio(status, "fits_open_file(template for FITS output)");
  std::unique_ptr<fitsfile, CloseFitsFile> close_template(tin);

  int hdutype = 0;
  fits_movabs_hdu(tin, 1, &hdutype, &status);
  throw_cfitsio(status, "fits_movabs_hdu(template primary HDU)");

  std::string outp = output_path_raw;
  if (outp.empty() || outp[0] != '!') outp = "!" + outp;
  std::vector<char> outp_path(outp.begin(), outp.end());
  outp_path.push_back('\0');

  fitsfile* outf = nullptr;
  fits_create_file(&outf, outp_path.data(), &status);
  throw_cfitsio(status, "fits_create_file(output FITS)");
  std::unique_ptr<fitsfile, CloseFitsFile> close_out(outf);

  fits_copy_header(tin, outf, &status);
  throw_cfitsio(status, "fits_copy_header(from template model FITS)");

  close_template.reset();

  apply_output_image_keys(outf, naxis1, naxis2, bunit, niter, crval1, crval2,
                          radesys_in, equinox, update_pointing_keywords, status);

  fits_write_img(outf, TFLOAT, 1, elements, write_ptr, &status);
  throw_cfitsio(status, "fits_write_img");
}

static void fill_fits_header_keys_from_fp(FitsHeader& h, ::fitsfile* fp) {
  int status = 0;
  fits_read_key(fp, TDOUBLE, "CDELT1", &h.cdelt1, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "CDELT2", &h.cdelt2, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "CRVAL1", &h.crval1, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "CRVAL2", &h.crval2, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "CRPIX1", &h.crpix1, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "CRPIX2", &h.crpix2, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "BMAJ", &h.beam_maj, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "BMIN", &h.beam_min, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TDOUBLE, "BPA", &h.beam_pa, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TFLOAT, "NOISE", &h.noise_keyword, nullptr, &status);
  if (status) status = 0;
  fits_read_key(fp, TFLOAT, "EQUINOX", &h.equinox, nullptr, &status);
  if (status) status = 0;
  int radesys_len = 0;
  fits_get_key_strlen(fp, "RADESYS", &radesys_len, &status);
  if (!status && radesys_len > 0) {
    std::vector<char> buf(radesys_len + 1, 0);
    fits_read_key(fp, TSTRING, "RADESYS", buf.data(), nullptr, &status);
    if (!status) h.radesys = buf.data();
  }
}

FitsHeader read_fits_header(const std::string& path) {
  FitsHeader h;
  try {
    CCfits::FITS fits(path, CCfits::Read, false);
    CCfits::PHDU& phdu = fits.pHDU();
    phdu.readAllKeys();
    h.naxis1 = phdu.axis(0);
    h.naxis2 = phdu.axis(1);
    h.bitpix = phdu.bitpix();
    fill_fits_header_keys_from_fp(h, fits.fitsPointer());
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
  return h;
}

FitsFloatImage read_fits_float_image(const std::string& path) {
  FitsFloatImage out;
  try {
    CCfits::FITS fits(path, CCfits::Read, true);
    CCfits::PHDU& phdu = fits.pHDU();
    phdu.readAllKeys();
    out.header.naxis1 = phdu.axis(0);
    out.header.naxis2 = phdu.axis(1);
    out.header.bitpix = phdu.bitpix();
    ::fitsfile* fp = fits.fitsPointer();
    fill_fits_header_keys_from_fp(out.header, fp);
    if (out.header.naxis1 <= 0 || out.header.naxis2 <= 0)
      throw std::runtime_error("read_fits_float_image: invalid dimensions");
    const long elements = out.header.naxis1 * out.header.naxis2;
    out.pixels.resize(static_cast<size_t>(elements));
    float nullval = 0.0f;
    int anynul = 0, status = 0;
    fits_read_img(fp, TFLOAT, 1, elements, &nullval, out.pixels.data(), &anynul, &status);
    if (status) throw std::runtime_error("read_fits_float_image: fits_read_img failed");
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
  return out;
}

std::vector<float> read_fits_image_float(const std::string& path) {
  return read_fits_float_image(path).pixels;
}

std::vector<double> read_fits_image_double(const std::string& path) {
  FitsHeader h = read_fits_header(path);
  if (h.naxis1 <= 0 || h.naxis2 <= 0)
    throw std::runtime_error("read_fits_image_double: invalid dimensions");
  try {
    CCfits::FITS fits(path, CCfits::Read, true);
    ::fitsfile* fp = fits.fitsPointer();
    const long elements = h.naxis1 * h.naxis2;
    std::vector<double> data(static_cast<size_t>(elements));
    double nullval = 0.0;
    int anynul = 0, status = 0;
    fits_read_img(fp, TDOUBLE, 1, elements, &nullval, data.data(), &anynul, &status);
    if (status)
      throw std::runtime_error("read_fits_image_double: fits_read_img failed");
    return data;
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
  return {};
}

std::vector<int> read_fits_image_int(const std::string& path) {
  FitsHeader h = read_fits_header(path);
  if (h.naxis1 <= 0 || h.naxis2 <= 0)
    throw std::runtime_error("read_fits_image_int: invalid dimensions");
  try {
    CCfits::FITS fits(path, CCfits::Read, true);
    ::fitsfile* fp = fits.fitsPointer();
    const long elements = h.naxis1 * h.naxis2;
    std::vector<int> data(static_cast<size_t>(elements));
    int nullval = 0;
    int anynul = 0, status = 0;
    fits_read_img(fp, TINT, 1, elements, &nullval, data.data(), &anynul, &status);
    if (status)
      throw std::runtime_error("read_fits_image_int: fits_read_img failed");
    return data;
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
  return {};
}

void write_fits_image_slice(const WriteFitsImageOptions& opts) {
  if (!opts.data || opts.naxis1 <= 0 || opts.naxis2 <= 0)
    throw std::runtime_error("write_fits_image_slice: invalid data or dimensions");
  const long elements = opts.naxis1 * opts.naxis2;
  const long offset = opts.naxis1 * opts.naxis2 * opts.plane_index;
  std::vector<float> host_buf;
  const float* write_ptr = nullptr;
  if (opts.data_on_device) {
    host_buf.resize(static_cast<size_t>(elements));
    checkCudaErrors(cudaMemcpy(host_buf.data(), opts.data + offset,
                               elements * sizeof(float), cudaMemcpyDeviceToHost));
    write_ptr = host_buf.data();
  } else {
    write_ptr = opts.data + offset;
  }
  std::vector<float> scaled_buf;
  if (opts.normalize && opts.normalization_factor != 1.0f) {
    scaled_buf.resize(static_cast<size_t>(elements));
    for (long i = 0; i < elements; i++)
      scaled_buf[static_cast<size_t>(i)] =
          write_ptr[static_cast<size_t>(i)] * opts.normalization_factor;
    write_ptr = scaled_buf.data();
  }
  std::string out_path = opts.output_path;
  if (out_path.empty() || out_path[0] != '!') out_path = "!" + out_path;
  if (opts.inline_primary_header) {
    const FitsHeader wire = ::gpuvmem::fits_wire_header_from_imaging(
        *opts.inline_primary_header, opts.naxis1, opts.naxis2);
    write_float_image_from_inline_header_cfitsio(
        wire, out_path, opts.naxis1, opts.naxis2, elements,
        const_cast<float*>(write_ptr), opts.bunit ? opts.bunit : "", opts.niter,
        opts.crval1, opts.crval2, opts.radesys, opts.equinox, true);
  } else if (!opts.header_template.empty()) {
    write_float_image_from_template_cfitsio(
        opts.header_template, out_path, opts.naxis1, opts.naxis2, elements,
        const_cast<float*>(write_ptr), opts.bunit ? opts.bunit : "", opts.niter,
        opts.crval1, opts.crval2, opts.radesys, opts.equinox, true);
  } else {
    throw std::runtime_error(
        "write_fits_image_slice: need inline_primary_header or non-empty header_template");
  }
}

void write_fits_image_complex(const WriteFitsComplexImageOptions& opts) {
  if (!opts.data || opts.naxis1 <= 0 || opts.naxis2 <= 0)
    throw std::runtime_error("write_fits_image_complex: invalid data or dimensions");
  const long elements = opts.naxis1 * opts.naxis2;
  std::vector<cufftComplex> host_buf;
  const cufftComplex* read_ptr = nullptr;
  if (opts.data_on_device) {
    host_buf.resize(static_cast<size_t>(elements));
    checkCudaErrors(cudaMemcpy(host_buf.data(), opts.data,
                               elements * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    read_ptr = host_buf.data();
  } else {
    read_ptr = opts.data;
  }
  // Convert complex to float based on output_type
  std::vector<float> image2D(static_cast<size_t>(elements));
  for (long i = 0; i < elements; i++) {
    const cufftComplex& z = read_ptr[static_cast<size_t>(i)];
    switch (opts.output_type) {
      case WriteFitsComplexImageOptions::AMPLITUDE:
        image2D[static_cast<size_t>(i)] = std::sqrt(z.x * z.x + z.y * z.y);
        break;
      case WriteFitsComplexImageOptions::PHASE:
        image2D[static_cast<size_t>(i)] = std::atan2(z.y, z.x) * 180.0f / M_PI;
        break;
      case WriteFitsComplexImageOptions::REAL:
        image2D[static_cast<size_t>(i)] = z.x;
        break;
      case WriteFitsComplexImageOptions::IMAG:
        image2D[static_cast<size_t>(i)] = z.y;
        break;
    }
  }
  std::string out_path = opts.output_path;
  if (out_path.empty() || out_path[0] != '!') out_path = "!" + out_path;
  if (opts.inline_primary_header) {
    const FitsHeader wire = ::gpuvmem::fits_wire_header_from_imaging(
        *opts.inline_primary_header, opts.naxis1, opts.naxis2);
    write_float_image_from_inline_header_cfitsio(
        wire, out_path, opts.naxis1, opts.naxis2, elements,
        image2D.data(), opts.bunit ? opts.bunit : "JY/PIXEL", opts.niter, 0.0, 0.0, "",
        0.0f, false);
  } else if (!opts.header_template.empty()) {
    write_float_image_from_template_cfitsio(
        opts.header_template, out_path, opts.naxis1, opts.naxis2, elements,
        image2D.data(), opts.bunit ? opts.bunit : "JY/PIXEL", opts.niter, 0.0,
        0.0, "", 0.0f, false);
  } else {
    throw std::runtime_error(
        "write_fits_image_complex: need inline_primary_header or non-empty header_template");
  }
}

}  // namespace fits
}  // namespace gpuvmem
