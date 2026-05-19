#ifndef GPUVMEM_MS_METADATA_H
#define GPUVMEM_MS_METADATA_H

#include <cufft.h>
#include <cuda_runtime.h>

#include <array>
#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * Spectral Window: defines the frequency axis for one SPW (from MS SPECTRAL_WINDOW).
 * Stored once at MS level; shared by all fields that use this SPW.
 */
class SpectralWindow {
 public:
  SpectralWindow() = default;
  SpectralWindow(int spw_id, std::vector<double> frequencies)
      : spw_id_(spw_id), frequencies_(std::move(frequencies)) {}

  int spectral_window_id() const { return spw_id_; }
  void set_spectral_window_id(int id) { spw_id_ = id; }

  int nchan() const { return static_cast<int>(frequencies_.size()); }
  const std::vector<double>& frequencies() const { return frequencies_; }
  void set_frequencies(std::vector<double> frequencies) {
    frequencies_ = std::move(frequencies);
  }
  double frequency(int chan) const { return frequencies_.at(chan); }

 private:
  int spw_id_{0};
  std::vector<double> frequencies_;
};

/**
 * Polarization: one row from MS POLARIZATION table (per POLARIZATION_ID).
 * Stores NUM_CORR and CORR_TYPE for Stokes/correlation selection and conversion.
 * CORR_TYPE values: 1=RR, 2=LL, 3=RL, 4=LR (circular); 5=XX, 6=YY, 9=XY, 10=YX (linear).
 */
class Polarization {
 public:
  Polarization() = default;
  Polarization(int pol_id, int num_corr, std::vector<int> corr_type)
      : pol_id_(pol_id),
        num_corr_(num_corr),
        corr_type_(std::move(corr_type)) {}

  int polarization_id() const { return pol_id_; }
  int num_corr() const { return num_corr_; }
  const std::vector<int>& corr_type() const { return corr_type_; }

  void set_polarization_id(int id) { pol_id_ = id; }
  void set_num_corr(int n) { num_corr_ = n; }
  void set_corr_type(std::vector<int> corr_type) {
    corr_type_ = std::move(corr_type);
  }

 private:
  int pol_id_{0};
  int num_corr_{0};
  std::vector<int> corr_type_;
};

/**
 * Data Description: one (SPW, Polarization) from MS DATA_DESCRIPTION table.
 * Links to one SpectralWindow and one Polarization (npol). Stored once at MS level.
 * When Stokes/correlation selection is used at read time, selected_pol_indices
 * holds the MS pol indices that were read; npol is then the number read (not the MS NUM_CORR).
 */
class DataDescription {
 public:
  DataDescription() = default;
  DataDescription(int data_desc_id, int spw_id, int pol_id, int nchan, int npol)
      : data_desc_id_(data_desc_id),
        spw_id_(spw_id),
        pol_id_(pol_id),
        nchan_(nchan),
        npol_(npol) {}

  int data_desc_id() const { return data_desc_id_; }
  int spectral_window_id() const { return spw_id_; }
  int polarization_id() const { return pol_id_; }
  int nchan() const { return nchan_; }
  int npol() const { return npol_; }

  void set_data_desc_id(int id) { data_desc_id_ = id; }
  void set_spectral_window_id(int id) { spw_id_ = id; }
  void set_polarization_id(int id) { pol_id_ = id; }
  void set_nchan(int n) { nchan_ = n; }
  void set_npol(int n) { npol_ = n; }

  /** When non-empty, only these MS pol indices were read; stored pol = 0..size()-1. */
  const std::vector<int>& selected_pol_indices() const {
    return selected_pol_indices_;
  }
  void set_selected_pol_indices(std::vector<int> v) {
    selected_pol_indices_ = std::move(v);
  }
  bool has_pol_selection() const { return !selected_pol_indices_.empty(); }

 private:
  int data_desc_id_{0};
  int spw_id_{0};
  int pol_id_{0};
  int nchan_{0};
  int npol_{0};
  std::vector<int> selected_pol_indices_;
};

/**
 * Primary beam model type (matches legacy AIRYDISK / GAUSSIAN).
 */
enum class PrimaryBeamType { Gaussian = 0, AiryDisk = 1 };

/**
 * Antenna metadata from MS ANTENNA table: position, diameter, primary beam params.
 * Stored per antenna at MS level; used for PB calculation and baseline products.
 */
struct Antenna {
  std::string antenna_id;
  std::string station;
  double3 position{0.0, 0.0, 0.0};  // ITRF or telescope frame [m]
  float antenna_diameter{0.f};       // [m]
  float pb_factor{0.f};              // scaling (e.g. 1.22 for Airy, 1.25 for Gaussian)
  float pb_cutoff{0.f};              // angular radius [rad] beyond which PB = 0
  PrimaryBeamType primary_beam{PrimaryBeamType::Gaussian};
};

/**
 * Field metadata from MS FIELD table: phase center, reference center, etc.
 * Owned by Field; not shared.
 */
struct FieldMetadata {
  int field_id{0};
  std::string name;
  std::string code;
  std::array<double, 2> reference_dir{{0.0, 0.0}};  // ra, dec [rad]
  std::array<double, 2> phase_dir{{0.0, 0.0}};      // phase center [rad]
  // Imaging: often derived from phase_dir and grid
  float ref_xobs_pix{0.f};
  float ref_yobs_pix{0.f};
  float phs_xobs_pix{0.f};
  float phs_yobs_pix{0.f};
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_METADATA_H
