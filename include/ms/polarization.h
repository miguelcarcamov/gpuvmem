#ifndef GPUVMEM_MS_POLARIZATION_H
#define GPUVMEM_MS_POLARIZATION_H

#include "ms/measurement_set_metadata.h"
#include "ms/metadata.h"

#include <complex>
#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

class MeasurementSet;

/**
 * CORR_TYPE values from CASA POLARIZATION table.
 * Circular: 1=RR, 2=LL, 3=RL, 4=LR.
 * Linear: 5=XX, 6=YY, 9=XY, 10=YX.
 */
namespace CorrType {
constexpr int RR = 1;
constexpr int LL = 2;
constexpr int RL = 3;
constexpr int LR = 4;
constexpr int XX = 5;
constexpr int YY = 6;
constexpr int XY = 9;
constexpr int YX = 10;
}  // namespace CorrType

/** Feed / correlation basis: linear, circular, or mixed. */
enum class FeedType { Linear, Circular, Mixed };

/** Stokes parameter names. */
constexpr const char* StokesI = "I";
constexpr const char* StokesQ = "Q";
constexpr const char* StokesU = "U";
constexpr const char* StokesV = "V";

/**
 * Returns the correlation name for a CASA CORR_TYPE (e.g. 5 -> "XX").
 * Returns empty string for unknown types.
 */
std::string correlation_name(int corr_type);

/**
 * Returns true if corr_type is a circular correlation (RR, LL, RL, LR).
 */
bool is_circular(int corr_type);

/**
 * Returns true if corr_type is a linear correlation (XX, YY, XY, YX).
 */
bool is_linear(int corr_type);

/**
 * Polarization helper: given CORR_TYPE (or Polarization metadata), computes
 * feed type, available Stokes, and complex conversion matrices per pol_id
 * (or per data_desc_id via metadata).
 *
 * Conversion matrices are complex, row-major: rows = Stokes (or correlations
 * for inverse), columns = correlations (or Stokes for inverse). Used for
 * correlations <-> Stokes conversion with proper weight propagation
 * (variance: sigma²_S = sum |a_i|² sigma²_i).
 */
class PolarizationHelper {
 public:
  using Complex = std::complex<float>;
  /** Row-major matrix: mat[row][col], e.g. mat[stokes_idx][corr_idx]. */
  using Matrix = std::vector<std::vector<Complex>>;

  explicit PolarizationHelper(const MeasurementSetMetadata* metadata)
      : metadata_(metadata) {}

  /** Feed type for this polarization (linear, circular, or mixed). */
  FeedType feed_type(int pol_id) const;

  /** Feed type for the polarization attached to this data description. */
  FeedType feed_type_for_data_desc(int data_desc_id) const;

  /** Correlation names in CORR_TYPE order (e.g. ["XX","YY","XY","YX"]). */
  std::vector<std::string> correlation_names(int pol_id) const;

  /**
   * Stokes parameters that can be formed from this polarization's
   * correlations (e.g. ["I","Q","U","V"] for full linear/circular).
   */
  std::vector<std::string> available_stokes(int pol_id) const;

  /** Available Stokes for the polarization of this data description. */
  std::vector<std::string> available_stokes_for_data_desc(
      int data_desc_id) const;

  /**
   * Matrix M such that S = M * C (Stokes = M * correlations).
   * Rows correspond to stokes_list (e.g. ["I","Q","U","V"]), columns to
   * correlation order in the polarization. Only stokes that can be formed
   * from the present correlations are included; stokes_list is filtered
   * to available_stokes if needed.
   */
  Matrix get_correlation_to_stokes_matrix(
      int pol_id, const std::vector<std::string>& stokes_list) const;

  Matrix get_correlation_to_stokes_matrix_for_data_desc(
      int data_desc_id, const std::vector<std::string>& stokes_list) const;

  /**
   * Matrix M such that C = M * S (correlations = M * Stokes).
   * Rows = correlation order, columns = stokes_list. Used for Stokes -> correlations.
   */
  Matrix get_stokes_to_correlation_matrix(
      int pol_id, const std::vector<std::string>& stokes_list) const;

  Matrix get_stokes_to_correlation_matrix_for_data_desc(
      int data_desc_id, const std::vector<std::string>& stokes_list) const;

 private:
  const MeasurementSetMetadata* metadata_{nullptr};

  /** Build correlation -> Stokes matrix for full circular (4 corr). */
  static Matrix build_corr_to_stokes_circular(
      const std::vector<int>& corr_type);
  /** Build correlation -> Stokes matrix for full linear (4 corr). */
  static Matrix build_corr_to_stokes_linear(const std::vector<int>& corr_type);
  /** Build Stokes -> correlation matrix for full circular. */
  static Matrix build_stokes_to_corr_circular(
      const std::vector<int>& corr_type);
  /** Build Stokes -> correlation matrix for full linear. */
  static Matrix build_stokes_to_corr_linear(const std::vector<int>& corr_type);
  /** Extract sub-matrix for requested Stokes and present correlations. */
  static Matrix select_corr_to_stokes(const Matrix& full_c2s,
                                     const std::vector<int>& corr_type,
                                     const std::vector<std::string>& stokes);
  static Matrix select_stokes_to_corr(const Matrix& full_s2c,
                                     const std::vector<int>& corr_type,
                                     const std::vector<std::string>& stokes);
};

/**
 * Convert visibility data from correlations to Stokes (in-place).
 * Weight propagation: sigma²_S = sum |a_i|² sigma²_i, w_S = 1/sigma²_S.
 * Sets ms.storage_mode(Stokes) and ms.stored_stokes_list(stokes_list).
 * Returns true on success.
 */
bool correlations_to_stokes(MeasurementSet& ms,
                            const std::vector<std::string>& stokes_list);

/**
 * Convert visibility data from Stokes back to correlations (in-place).
 * Uses ms.stored_stokes_list() for the inverse matrix.
 * Required before writing to MS (writer requires correlation mode).
 * Returns true on success.
 */
bool stokes_to_correlations(MeasurementSet& ms);

/**
 * Returns true iff every data description in meta can form all requested Stokes
 * (i.e. each requested Stokes is in available_stokes_for_data_desc for every DD).
 * Use after reading MS to validate that imaging for requested_stokes is possible.
 * If false, the dataset cannot form one or more of the requested Stokes (e.g. user
 * asked for "I,Q,U,V" but the MS only has RR,LL giving I,V).
 */
bool stokes_supported_by_metadata(const MeasurementSetMetadata& meta,
                                  const std::vector<std::string>& requested_stokes);

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_POLARIZATION_H
