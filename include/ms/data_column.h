#ifndef GPUVMEM_MS_DATA_COLUMN_H
#define GPUVMEM_MS_DATA_COLUMN_H

#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * MS visibility data column identifiers.
 * DATA, CORRECTED_DATA, MODEL_DATA correspond to MAIN table columns.
 * RESIDUAL is derived (e.g. DATA - MODEL_DATA) and has no persistent column.
 */
enum class DataColumn {
  DATA,
  CORRECTED_DATA,
  MODEL_DATA,
  RESIDUAL,
};

/** True if this column exists as a MAIN table column (can be read/written). */
inline bool has_ms_column(DataColumn col) {
  return col != DataColumn::RESIDUAL;
}

/** MAIN table column name for reading/writing. RESIDUAL has no name. */
inline const char* ms_column_name(DataColumn col) {
  switch (col) {
    case DataColumn::DATA:
      return "DATA";
    case DataColumn::CORRECTED_DATA:
      return "CORRECTED_DATA";
    case DataColumn::MODEL_DATA:
      return "MODEL_DATA";
    case DataColumn::RESIDUAL:
      return nullptr;
  }
  return nullptr;
}

/** All columns that have a MAIN table column. */
inline std::vector<DataColumn> ms_stored_columns() {
  return {DataColumn::DATA, DataColumn::CORRECTED_DATA, DataColumn::MODEL_DATA};
}

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_DATA_COLUMN_H
