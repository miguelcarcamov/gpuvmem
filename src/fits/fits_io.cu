/* Thin C++ FITS I/O over CFITSIO. RAII, options struct, multi-plane aware.
 * See docs/fits_io_refactor.md. Implementation can later be swapped to CCfits.
 */

#include "fits/fits_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fitsio.h>
#include <helper_cuda.h>
#include <stdexcept>

namespace gpuvmem {
namespace fits {

namespace {

// RAII: close FITS file on scope exit.
class FitsFileGuard {
 public:
  explicit FitsFileGuard(::fitsfile* fp) : fp_(fp) {}
  ~FitsFileGuard() {
    if (fp_) {
      int status = 0;
      fits_close_file(fp_, &status);
      (void)status;
    }
  }
  FitsFileGuard(FitsFileGuard&& other) noexcept : fp_(other.fp_) {
    other.fp_ = nullptr;
  }
  FitsFileGuard& operator=(FitsFileGuard&& other) noexcept {
    if (this != &other) {
      if (fp_) {
        int status = 0;
        fits_close_file(fp_, &status);
      }
      fp_ = other.fp_;
      other.fp_ = nullptr;
    }
    return *this;
  }
  FitsFileGuard(const FitsFileGuard&) = delete;
  FitsFileGuard& operator=(const FitsFileGuard&) = delete;

  ::fitsfile* get() const { return fp_; }
  ::fitsfile* release() {
    ::fitsfile* p = fp_;
    fp_ = nullptr;
    return p;
  }

 private:
  ::fitsfile* fp_;
};

void throw_on_error(int status, const char* context) {
  if (status) {
    char err[FLEN_ERRMSG];
    fits_get_errstatus(status, err);
    char msg[512];
    snprintf(msg, sizeof(msg), "%s: CFITSIO error %d: %s", context, status, err);
    throw std::runtime_error(msg);
  }
}

}  // namespace

FitsHeader read_fits_header(const std::string& path) {
  ::fitsfile* fp = nullptr;
  int status = 0;
  fits_open_file(&fp, path.c_str(), READONLY, &status);
  throw_on_error(status, "read_fits_header: open");
  FitsFileGuard guard(fp);

  FitsHeader h;
  fits_read_key(fp, TDOUBLE, "CDELT1", &h.cdelt1, nullptr, &status);
  if (status) status = 0;  // optional
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
  fits_read_key(fp, TLONG, "NAXIS1", &h.naxis1, nullptr, &status);
  throw_on_error(status, "read_fits_header: NAXIS1");
  fits_read_key(fp, TLONG, "NAXIS2", &h.naxis2, nullptr, &status);
  throw_on_error(status, "read_fits_header: NAXIS2");
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
  status = 0;
  fits_get_img_type(fp, &h.bitpix, &status);
  if (status) status = 0;

  return h;
}

std::vector<float> read_fits_image_float(const std::string& path) {
  FitsHeader h = read_fits_header(path);
  if (h.naxis1 <= 0 || h.naxis2 <= 0)
    throw std::runtime_error("read_fits_image_float: invalid dimensions");
  size_t nelem = static_cast<size_t>(h.naxis1) * static_cast<size_t>(h.naxis2);

  ::fitsfile* fp = nullptr;
  int status = 0;
  fits_open_file(&fp, path.c_str(), READONLY, &status);
  throw_on_error(status, "read_fits_image_float: open");
  FitsFileGuard guard(fp);

  std::vector<float> data(nelem);
  long fpixel = 1;
  int anynull = 0;
  float nullval = 0;
  fits_read_img(fp, TFLOAT, fpixel, static_cast<long>(nelem), &nullval,
                data.data(), &anynull, &status);
  throw_on_error(status, "read_fits_image_float: read_img");
  return data;
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
                               elements * sizeof(float),
                               cudaMemcpyDeviceToHost));
    write_ptr = host_buf.data();
  } else {
    write_ptr = opts.data + offset;
  }

  // Optional normalize (normalization_factor)
  std::vector<float> scaled_buf;
  if (opts.normalize && opts.normalization_factor != 1.0f) {
    scaled_buf.resize(static_cast<size_t>(elements));
    for (long i = 0; i < elements; i++)
      scaled_buf[static_cast<size_t>(i)] =
          write_ptr[static_cast<size_t>(i)] * opts.normalization_factor;
    write_ptr = scaled_buf.data();
  }

  ::fitsfile* template_fp = nullptr;
  int status = 0;
  fits_open_file(&template_fp, opts.header_template.c_str(), READONLY, &status);
  throw_on_error(status, "write_fits_image_slice: open template");
  FitsFileGuard template_guard(template_fp);

  ::fitsfile* out_fp = nullptr;
  std::string out_path = opts.output_path;
  if (out_path.empty() || out_path[0] != '!')
    out_path = "!" + out_path;  // overwrite
  fits_create_file(&out_fp, out_path.c_str(), &status);
  throw_on_error(status, "write_fits_image_slice: create");
  FitsFileGuard out_guard(out_fp);

  fits_copy_header(template_fp, out_fp, &status);
  throw_on_error(status, "write_fits_image_slice: copy_header");

  char bunit_buf[32] = "";
  if (opts.bunit) snprintf(bunit_buf, sizeof(bunit_buf), "%s", opts.bunit);
  fits_update_key(out_fp, TSTRING, "BUNIT", bunit_buf,
                  "Unit of measurement", &status);
  if (status) status = 0;
  int niter_val = opts.niter;
  fits_update_key(out_fp, TINT, "NITER", &niter_val,
                  "Number of iteration in gpuvmem software", &status);
  if (status) status = 0;
  long na1 = opts.naxis1, na2 = opts.naxis2;
  fits_update_key(out_fp, TINT, "NAXIS1", &na1, "", &status);
  if (status) status = 0;
  fits_update_key(out_fp, TINT, "NAXIS2", &na2, "", &status);
  if (status) status = 0;
  std::string radesys_str = opts.radesys;
  if (radesys_str.empty()) radesys_str = "ICRS";
  fits_update_key(out_fp, TSTRING, "RADESYS",
                  radesys_str.c_str(), "Changed by gpuvmem", &status);
  if (status) status = 0;
  float equinox_val = opts.equinox;
  fits_update_key(out_fp, TFLOAT, "EQUINOX", &equinox_val,
                  "Changed by gpuvmem", &status);
  if (status) status = 0;
  double crval1_val = opts.crval1, crval2_val = opts.crval2;
  fits_update_key(out_fp, TDOUBLE, "CRVAL1", &crval1_val,
                  "Changed by gpuvmem", &status);
  if (status) status = 0;
  fits_update_key(out_fp, TDOUBLE, "CRVAL2", &crval2_val,
                  "Changed by gpuvmem", &status);
  if (status) status = 0;

  long fpixel = 1;
  fits_write_img(out_fp, TFLOAT, fpixel, elements,
                 const_cast<float*>(write_ptr), &status);
  throw_on_error(status, "write_fits_image_slice: write_img");
}

}  // namespace fits
}  // namespace gpuvmem
