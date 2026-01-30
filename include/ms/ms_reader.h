#ifndef GPUVMEM_MS_MS_READER_H
#define GPUVMEM_MS_MS_READER_H

#include "ms/data_column.h"
#include "ms/measurement_set.h"

#include <memory>
#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * Options for reading an MS into the visibility model.
 * Specifies which data column(s) to read and optional filters.
 */
struct MSReadOptions {
  /** Primary column to read into Vo (observed). Prefer CORRECTED_DATA if present. */
  DataColumn data_column{DataColumn::CORRECTED_DATA};
  /** If true, also read MODEL_DATA into Vm when the column exists. */
  bool read_model{false};
  /** If true, also read CORRECTED_DATA when data_column != CORRECTED_DATA (e.g. for residual). */
  bool read_corrected{false};
  /** If true, also read raw DATA when different from data_column. */
  bool read_data{false};
  /** Random sampling fraction (1.0 = use all rows). */
  float random_probability{1.0f};
  /** If true, order by UVW[2] (W) when using W-projection. */
  bool order_by_w{false};
  /** If true, apply noise to read visibilities (simulation). */
  bool apply_noise{false};

  /**
   * Requested Stokes parameters to read (e.g. {"I"}, {"I","Q"}).
   * Empty = read all correlations (current behaviour).
   * Only correlations needed to form these Stokes are read; per-DD selection is applied.
   */
  std::vector<std::string> requested_stokes;
  /**
   * Requested correlations to read (e.g. {"XX","YY"}, {"RR","LL"}).
   * Empty = no correlation filter (or use requested_stokes if set).
   * If both requested_stokes and requested_correlations are set, requested_stokes takes precedence.
   */
  std::vector<std::string> requested_correlations;
};

/**
 * Reader interface: load MAIN table (and sub-tables) into MeasurementSet.
 * At least one data column (DATA, CORRECTED_DATA, or MODEL_DATA) is read;
 * RESIDUAL is never read from disk (it is computed).
 */
class MSReader {
 public:
  MSReader() = default;
  virtual ~MSReader() = default;

  /**
   * Read MS from path into out. Metadata (spectral windows, data descriptions,
   * fields) and visibility data for the selected columns are populated.
   * Returns true on success.
   */
  virtual bool read(const std::string& path,
                    MeasurementSet& out,
                    const MSReadOptions& options = MSReadOptions()) = 0;
};

/** Create the default MS reader (casacore-based). */
std::unique_ptr<MSReader> create_ms_reader();

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_MS_READER_H
