#ifndef GPUVMEM_MS_MS_WRITER_H
#define GPUVMEM_MS_MS_WRITER_H

#include "ms/data_column.h"
#include "ms/measurement_set.h"

#include <memory>
#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * Options for writing the visibility model to an MS.
 * Specifies which column(s) to write and how to compute values.
 */
struct MSWriteOptions {
  /** Column(s) to write. Each must have has_ms_column() true. */
  std::vector<DataColumn> write_columns{DataColumn::MODEL_DATA};
  /** If true, write residual (Vo - Vm) into a column; requires a target column name (e.g. custom). */
  bool write_residual{false};
  /** When writing residual, use this as the column name (created if missing). */
  std::string residual_column_name{"RESIDUAL_DATA"};
  /** If true, also update WEIGHT from the in-memory weights. */
  bool update_weights{true};
  /** Random sampling order (must match read order if used). */
  float random_probability{1.0f};
  /** If true, order by W when matching rows. */
  bool order_by_w{false};
};

/**
 * Writer interface: write MeasurementSet visibility data back to MAIN table.
 * At least one writable column (DATA, CORRECTED_DATA, or MODEL_DATA) is supported;
 * RESIDUAL can be written as a derived column (e.g. DATA - MODEL_DATA) or custom name.
 *
 * The dataset must be in correlation mode to write (MS stores correlations).
 * If storage_mode() == Stokes, call stokes_to_correlations(ms) first, then write.
 */
class MSWriter {
 public:
  MSWriter() = default;
  virtual ~MSWriter() = default;

  /**
   * Write ms to an existing MS at path. Only selected columns are updated.
   * Table must exist; columns are created if missing (where applicable).
   * Returns true on success.
   */
  virtual bool write(const std::string& path,
                     const MeasurementSet& ms,
                     const MSWriteOptions& options = MSWriteOptions()) = 0;
};

/** Create the default MS writer (casacore-based). */
std::unique_ptr<MSWriter> create_ms_writer();

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_MS_WRITER_H
