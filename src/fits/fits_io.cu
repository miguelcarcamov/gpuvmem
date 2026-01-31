/* FITS I/O in gpuvmem: CCfits (mandatory). RAII, options struct, multi-plane.
 * See docs/fits_io_refactor.md. CCfits wraps CFITSIO; we link both.
 */

#include "fits/fits_io.h"
#include <CCfits/FITS.h>
#include <CCfits/PHDU.h>
#include <cstdio>
#include <cstring>
#include <fitsio.h>
#include <helper_cuda.h>
#include <stdexcept>
#include <valarray>
#include <vector>

namespace gpuvmem {
namespace fits {

namespace {
void throw_from_ccfits(const CCfits::FitsException& e) {
  throw std::runtime_error(std::string("CCfits: ") + e.message());
}
}  // namespace

FitsHeader read_fits_header(const std::string& path) {
  FitsHeader h;
  try {
    CCfits::FITS fits(path, CCfits::Read, false);
    CCfits::PHDU& phdu = fits.pHDU();
    phdu.readAllKeys();
    h.naxis1 = phdu.axis(0);
    h.naxis2 = phdu.axis(1);
    h.bitpix = phdu.bitpix();
    ::fitsfile* fp = fits.fitsPointer();
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
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
  return h;
}

std::vector<float> read_fits_image_float(const std::string& path) {
  FitsHeader h = read_fits_header(path);
  if (h.naxis1 <= 0 || h.naxis2 <= 0)
    throw std::runtime_error("read_fits_image_float: invalid dimensions");
  try {
    CCfits::FITS fits(path, CCfits::Read, true);
    CCfits::PHDU& phdu = fits.pHDU();
    std::valarray<float> contents;
    phdu.read(contents);
    std::vector<float> data(contents.size());
    for (size_t i = 0; i < data.size(); i++)
      data[i] = contents[i];
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
  try {
    CCfits::FITS templateFits(opts.header_template, CCfits::Read, false);
    std::string out_path = opts.output_path;
    if (out_path.empty() || out_path[0] != '!')
      out_path = "!" + out_path;
    CCfits::FITS outFits(out_path, templateFits);
    ::fitsfile* fp = outFits.fitsPointer();
    int status = 0;
    char bunit_buf[32] = "";
    if (opts.bunit) snprintf(bunit_buf, sizeof(bunit_buf), "%s", opts.bunit);
    fits_update_key(fp, TSTRING, "BUNIT", bunit_buf, "Unit of measurement", &status);
    if (status) status = 0;
    int niter_val = opts.niter;
    fits_update_key(fp, TINT, "NITER", &niter_val,
                    "Number of iteration in gpuvmem software", &status);
    if (status) status = 0;
    long na1 = opts.naxis1, na2 = opts.naxis2;
    fits_update_key(fp, TINT, "NAXIS1", &na1, "", &status);
    if (status) status = 0;
    fits_update_key(fp, TINT, "NAXIS2", &na2, "", &status);
    if (status) status = 0;
    std::string radesys_str = opts.radesys;
    if (radesys_str.empty()) radesys_str = "ICRS";
    char radesys_buf[32];
    snprintf(radesys_buf, sizeof(radesys_buf), "%.31s", radesys_str.c_str());
    fits_update_key(fp, TSTRING, "RADESYS", radesys_buf, "Changed by gpuvmem", &status);
    if (status) status = 0;
    float equinox_val = opts.equinox;
    fits_update_key(fp, TFLOAT, "EQUINOX", &equinox_val, "Changed by gpuvmem", &status);
    if (status) status = 0;
    double crval1_val = opts.crval1, crval2_val = opts.crval2;
    fits_update_key(fp, TDOUBLE, "CRVAL1", &crval1_val, "Changed by gpuvmem", &status);
    if (status) status = 0;
    fits_update_key(fp, TDOUBLE, "CRVAL2", &crval2_val, "Changed by gpuvmem", &status);
    if (status) status = 0;
    std::valarray<float> arr(static_cast<size_t>(elements));
    for (long i = 0; i < elements; i++)
      arr[static_cast<size_t>(i)] = write_ptr[static_cast<size_t>(i)];
    outFits.pHDU().write(1, elements, arr);
  } catch (const CCfits::FitsException& e) {
    throw_from_ccfits(e);
  }
}

}  // namespace fits
}  // namespace gpuvmem
