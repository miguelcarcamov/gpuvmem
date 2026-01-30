/* MS Writer: write MeasurementSet back to MAIN table (casacore). */
#include "ms/ms_writer.h"
#include "ms/visibility_model.h"

#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/Vector.h>
#include <tables/TaQL/TableParse.h>
#include <tables/Tables/ArrColDesc.h>
#include <tables/Tables/ArrayColumn.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableDesc.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace gpuvmem {
namespace ms {

static bool table_has_column(const casacore::Table& tab, const std::string& name) {
  return tab.tableDesc().isColumn(name);
}

class CasacoreMSWriter : public MSWriter {
 public:
  bool write(const std::string& path,
             const MeasurementSet& ms,
             const MSWriteOptions& options) override {
    try {
      if (ms.storage_mode() == StorageMode::Stokes) {
        std::fprintf(stderr,
                     "MSWriter: dataset must be in correlation mode to write; "
                     "call stokes_to_correlations(ms) first.\n");
        return false;
      }
      casacore::Table main_tab(path, casacore::Table::Update);
      if (main_tab.nrow() == 0) {
        std::fprintf(stderr, "MSWriter: empty MAIN table\n");
        return false;
      }

      for (DataColumn col : options.write_columns) {
        if (!has_ms_column(col)) continue;
        const char* name = ms_column_name(col);
        if (!name) continue;
        ensure_column(main_tab, path, name);
        write_column(main_tab, path, ms, name, col, options);
      }
      if (options.write_residual && !options.residual_column_name.empty())
        write_residual_column(main_tab, path, ms, options);

      main_tab.flush();
      return true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MSWriter: %s\n", e.what());
      return false;
    }
  }

 private:
  void ensure_column(casacore::Table& main_tab,
                     const std::string& dir,
                     const std::string& column_name) {
    if (table_has_column(main_tab, column_name)) return;
    main_tab.addColumn(casacore::ArrayColumnDesc<casacore::Complex>(
        column_name, "created by gpuvmem"));
    std::string query = "UPDATE " + dir + " SET " + column_name + "=DATA";
    casacore::tableCommand(query.c_str());
    main_tab.flush();
  }

  void write_column(casacore::Table& main_tab,
                    const std::string& dir,
                    const MeasurementSet& ms,
                    const std::string& column_name,
                    DataColumn col,
                    const MSWriteOptions& options) {
    std::string query =
        "select WEIGHT," + column_name + ",FLAG from " + dir +
        " where !FLAG_ROW and ANY(!FLAG)";
    query += " ORDERBY FIELD_ID, ANTENNA1, ANTENNA2, TIME, DATA_DESC_ID";
    if (options.order_by_w) query += ", UVW[2]";
    casacore::Table query_tab = casacore::tableCommand(query.c_str());
    casacore::ArrayColumn<float> weight_col(query_tab, "WEIGHT");
    casacore::ArrayColumn<casacore::Complex> data_col(query_tab, column_name);
    casacore::ArrayColumn<bool> flag_col(query_tab, "FLAG");

    size_t row_idx = 0;
    for (size_t f = 0; f < ms.num_fields(); f++) {
      const Field& field = ms.field(f);
      for (const Baseline& bl : field.baselines()) {
        for (const TimeSample& ts : bl.time_samples()) {
          if (row_idx >= query_tab.nrow()) break;
          casacore::Vector<float> weights = weight_col(row_idx);
          casacore::Matrix<casacore::Complex> dataCol = data_col(row_idx);
          casacore::Matrix<bool> flagCol = flag_col(row_idx);
          size_t vi = 0;
          const auto& vis = ts.visibilities();
          for (int p = 0; p < dataCol.nrow() && vi < vis.size(); p++) {
            for (int j = 0; j < dataCol.ncolumn() && vi < vis.size(); j++) {
              if (flagCol(p, j) || weights(p) <= 0.0f) continue;
              cufftComplex v;
              if (col == DataColumn::MODEL_DATA) v = vis[vi].Vm;
              else if (col == DataColumn::CORRECTED_DATA || col == DataColumn::DATA) v = vis[vi].Vo;
              else v = vis[vi].Vr;
              dataCol(p, j) = casacore::Complex(v.x, v.y);
              vi++;
            }
          }
          data_col.put(row_idx, dataCol);
          if (options.update_weights && !vis.empty()) {
            int npol = weights.size();
            int nchan = dataCol.ncolumn();
            for (int p = 0; p < npol; p++) {
              float sum = 0.f;
              int n = 0;
              for (int j = 0; j < nchan && (j * npol + p) < static_cast<int>(vis.size()); j++) {
                sum += vis[j * npol + p].imaging_weight;
                n++;
              }
              weights(p) = (n > 0) ? (sum / n) : 1.f;
            }
            weight_col.put(row_idx, weights);
          }
          row_idx++;
        }
      }
    }
    query_tab.flush();
  }

  void write_residual_column(casacore::Table& main_tab,
                             const std::string& dir,
                             const MeasurementSet& ms,
                             const MSWriteOptions& options) {
    const std::string& col_name = options.residual_column_name;
    if (col_name.empty()) return;
    ensure_column(main_tab, dir, col_name);
    std::string query =
        "select WEIGHT," + col_name + ",FLAG from " + dir +
        " where !FLAG_ROW and ANY(!FLAG)";
    query += " ORDERBY FIELD_ID, ANTENNA1, ANTENNA2, TIME, DATA_DESC_ID";
    if (options.order_by_w) query += ", UVW[2]";
    casacore::Table query_tab = casacore::tableCommand(query.c_str());
    casacore::ArrayColumn<casacore::Complex> data_col(query_tab, col_name);
    casacore::ArrayColumn<bool> flag_col(query_tab, "FLAG");
    casacore::ArrayColumn<float> weight_col(query_tab, "WEIGHT");

    size_t row_idx = 0;
    for (size_t f = 0; f < ms.num_fields(); f++) {
      const Field& field = ms.field(f);
      for (const Baseline& bl : field.baselines()) {
        for (const TimeSample& ts : bl.time_samples()) {
          if (row_idx >= query_tab.nrow()) break;
          casacore::Matrix<casacore::Complex> dataCol = data_col(row_idx);
          casacore::Matrix<bool> flagCol = flag_col(row_idx);
          casacore::Vector<float> weights = weight_col(row_idx);
          size_t vi = 0;
          const auto& vis = ts.visibilities();
          for (int p = 0; p < dataCol.nrow() && vi < vis.size(); p++) {
            for (int j = 0; j < dataCol.ncolumn() && vi < vis.size(); j++) {
              if (flagCol(p, j) || weights(p) <= 0.0f) continue;
              dataCol(p, j) = casacore::Complex(vis[vi].Vr.x, vis[vi].Vr.y);
              vi++;
            }
          }
          data_col.put(row_idx, dataCol);
          row_idx++;
        }
      }
    }
    query_tab.flush();
  }
};

std::unique_ptr<MSWriter> create_ms_writer() {
  return std::make_unique<CasacoreMSWriter>();
}

}  // namespace ms
}  // namespace gpuvmem
