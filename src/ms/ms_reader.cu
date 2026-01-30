/* MS Reader: load MAIN table into MeasurementSet (casacore). */
#include "ms/ms_reader.h"
#include "ms/polarization.h"
#include "ms/visibility_model.h"

#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/Vector.h>
#include <tables/TaQL/TableParse.h>
#include <tables/Tables/ArrayColumn.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/Table.h>

#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace gpuvmem {
namespace ms {

static std::string column_name_for_read(DataColumn col) {
  const char* n = ms_column_name(col);
  return n ? std::string(n) : std::string();
}

static bool table_has_column(const casacore::Table& tab, const std::string& name) {
  return tab.tableDesc().isColumn(name);
}

constexpr float LIGHTSPEED_MS = 2.99792458e8f;
constexpr float PB_DEFAULT_FACTOR = 1.2197f;  // first zero of J1 / pi

/** Apply requested_stokes or requested_correlations: set per-DD selected_pol_indices and npol. */
static void apply_pol_selection(const MSReadOptions& options,
                                MeasurementSet& out) {
  if (options.requested_stokes.empty() && options.requested_correlations.empty())
    return;
  PolarizationHelper helper(&out.metadata());
  for (const auto& dd : out.metadata().data_descriptions()) {
    int dd_id = dd.data_desc_id();
    DataDescription* pdd = out.metadata().find_data_description(dd_id);
    if (!pdd) continue;
    std::vector<int> selected;
    if (!options.requested_stokes.empty()) {
      auto M = helper.get_correlation_to_stokes_matrix_for_data_desc(
          dd_id, options.requested_stokes);
      std::set<int> cols_needed;
      for (const auto& row : M) {
        for (size_t c = 0; c < row.size(); c++) {
          if (row[c].real() != 0.f || row[c].imag() != 0.f)
            cols_needed.insert(static_cast<int>(c));
        }
      }
      selected.assign(cols_needed.begin(), cols_needed.end());
    } else {
      int pol_id = pdd->polarization_id();
      std::vector<std::string> names = helper.correlation_names(pol_id);
      const std::set<std::string> want(options.requested_correlations.begin(),
                                        options.requested_correlations.end());
      for (size_t i = 0; i < names.size(); i++) {
        if (want.count(names[i])) selected.push_back(static_cast<int>(i));
      }
    }
    if (!selected.empty()) {
      pdd->set_selected_pol_indices(std::move(selected));
      pdd->set_npol(static_cast<int>(pdd->selected_pol_indices().size()));
    }
  }
}

static void read_antennas(const std::string& dir,
                          casacore::Table& main_tab,
                          MeasurementSet& out) {
  std::string ant_query =
      "select POSITION,DISH_DIAMETER,NAME,STATION FROM " + dir +
      "/ANTENNA where !FLAG_ROW";
  casacore::Table antenna_tab(casacore::tableCommand(ant_query.c_str()));
  if (antenna_tab.nrow() == 0) return;

  std::string telescope_name;
  try {
    casacore::Table obs_tab(dir + "/OBSERVATION");
    if (obs_tab.nrow() > 0 && obs_tab.tableDesc().isColumn("TELESCOPE_NAME")) {
      casacore::ROScalarColumn<casacore::String> tel_col(obs_tab, "TELESCOPE_NAME");
      telescope_name = tel_col(0);
    }
  } catch (...) {
  }

  float min_freq_hz = 1e9f;
  for (const auto& spw : out.metadata().spectral_windows()) {
    for (int c = 0; c < spw.nchan(); c++) {
      float f = static_cast<float>(spw.frequency(c));
      if (f > 0.f && f < min_freq_hz) min_freq_hz = f;
    }
  }
  if (min_freq_hz <= 0.f) min_freq_hz = 1e9f;
  float max_wavelength = LIGHTSPEED_MS / min_freq_hz;

  casacore::ROArrayColumn<double> pos_col(antenna_tab, "POSITION");
  casacore::ROScalarColumn<double> dish_col(antenna_tab, "DISH_DIAMETER");
  casacore::ROScalarColumn<casacore::String> name_col(antenna_tab, "NAME");
  casacore::ROScalarColumn<casacore::String> station_col(antenna_tab, "STATION");

  for (size_t a = 0; a < antenna_tab.nrow(); a++) {
    Antenna ant;
    ant.antenna_id = name_col(a);
    ant.station = station_col(a);
    casacore::Vector<double> pos = pos_col(a);
    ant.position.x = pos.size() > 0 ? pos(0) : 0.0;
    ant.position.y = pos.size() > 1 ? pos(1) : 0.0;
    ant.position.z = pos.size() > 2 ? pos(2) : 0.0;
    ant.antenna_diameter = static_cast<float>(dish_col(a));

    if (telescope_name == "ALMA") {
      ant.pb_factor = 1.13f;
      ant.primary_beam = PrimaryBeamType::AiryDisk;
    } else if (telescope_name == "EVLA") {
      ant.pb_factor = 1.25f;
      ant.primary_beam = PrimaryBeamType::Gaussian;
    } else {
      ant.pb_factor = PB_DEFAULT_FACTOR;
      ant.primary_beam = PrimaryBeamType::Gaussian;
    }
    ant.pb_cutoff = (ant.antenna_diameter > 0.f)
                        ? (ant.pb_factor * max_wavelength / ant.antenna_diameter)
                        : 0.f;
    out.metadata().add_antenna(std::move(ant));
  }
}

class CasacoreMSReader : public MSReader {
 public:
  bool read(const std::string& path,
            MeasurementSet& out,
            const MSReadOptions& options) override {
    try {
      casacore::Table main_tab(path);
      if (main_tab.nrow() == 0) {
        std::fprintf(stderr, "MSReader: empty MAIN table\n");
        return false;
      }

      std::string data_col = column_name_for_read(options.data_column);
      if (data_col.empty()) {
        std::fprintf(stderr, "MSReader: RESIDUAL cannot be read from disk\n");
        return false;
      }
      if (!table_has_column(main_tab, data_col)) {
        if (table_has_column(main_tab, "CORRECTED_DATA")) data_col = "CORRECTED_DATA";
        else if (table_has_column(main_tab, "DATA")) data_col = "DATA";
        else {
          std::fprintf(stderr, "MSReader: no DATA/CORRECTED_DATA column\n");
          return false;
        }
      }

      out.set_name(path);
      read_metadata(main_tab, path, out);
      apply_pol_selection(options, out);
      read_visibilities(main_tab, path, data_col, options, out);
      return true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MSReader: %s\n", e.what());
      return false;
    }
  }

 private:
  void read_metadata(casacore::Table& main_tab,
                     const std::string& dir,
                     MeasurementSet& out) {
    std::string field_query =
        "select REFERENCE_DIR,PHASE_DIR,ROWID() AS ID FROM " + dir +
        "/FIELD where !FLAG_ROW";
    casacore::Table field_tab(casacore::tableCommand(field_query.c_str()));

    std::string aux_spw = "select SPECTRAL_WINDOW_ID FROM " + dir +
                          "/DATA_DESCRIPTION where !FLAG_ROW";
    std::string spw_query =
        "select NUM_CHAN,CHAN_FREQ,ROWID() as ID FROM " + dir +
        "/SPECTRAL_WINDOW where !FLAG_ROW AND ROWID() in [" + aux_spw + "]";
    casacore::Table spw_tab(casacore::tableCommand(spw_query.c_str()));

    std::string aux_pol = "select POLARIZATION_ID FROM " + dir +
                          "/DATA_DESCRIPTION where !FLAG_ROW";
    std::string pol_query =
        "select NUM_CORR,CORR_TYPE,ROWID() as ID from " + dir +
        "/POLARIZATION where !FLAG_ROW and ROWID() in [" + aux_pol + "]";
    casacore::Table pol_tab(casacore::tableCommand(pol_query.c_str()));

    casacore::ROScalarColumn<casacore::Int64> nchan_col(spw_tab, "NUM_CHAN");
    casacore::ROArrayColumn<double> chan_freq_col(spw_tab, "CHAN_FREQ");
    casacore::ROScalarColumn<casacore::Int64> spw_id_col(spw_tab, "ID");

    int nspw = static_cast<int>(spw_tab.nrow());
    for (int i = 0; i < nspw; i++) {
      int spw_id = static_cast<int>(spw_id_col(i));
      int nchan = static_cast<int>(nchan_col(i));
      casacore::Vector<double> freq = chan_freq_col(i);
      std::vector<double> frequencies(freq.size());
      for (int j = 0; j < nchan; j++) frequencies[j] = freq(j);
      out.metadata().add_spectral_window(
          SpectralWindow(spw_id, std::move(frequencies)));
    }

    std::string dd_query =
        "select SPECTRAL_WINDOW_ID,POLARIZATION_ID,ROWID() as ID FROM " + dir +
        "/DATA_DESCRIPTION where !FLAG_ROW";
    casacore::Table dd_tab(casacore::tableCommand(dd_query.c_str()));
    casacore::ROScalarColumn<casacore::Int64> dd_spw_col(dd_tab, "SPECTRAL_WINDOW_ID");
    casacore::ROScalarColumn<casacore::Int64> dd_pol_col(dd_tab, "POLARIZATION_ID");
    casacore::ROScalarColumn<casacore::Int64> dd_id_col(dd_tab, "ID");

    if (pol_tab.nrow() > 0) {
      casacore::ROScalarColumn<casacore::Int64> ncorr_col(pol_tab, "NUM_CORR");
      casacore::ROScalarColumn<casacore::Int64> pol_id_col(pol_tab, "ID");
      const bool has_corr_type = pol_tab.tableDesc().isColumn("CORR_TYPE");
      std::unique_ptr<casacore::ROArrayColumn<casacore::Int>> corr_type_col_ptr;
      if (has_corr_type) {
        corr_type_col_ptr =
            std::make_unique<casacore::ROArrayColumn<casacore::Int>>(pol_tab,
                                                                   "CORR_TYPE");
      }
      for (size_t r = 0; r < pol_tab.nrow(); r++) {
        int pol_id = static_cast<int>(pol_id_col(r));
        int num_corr = static_cast<int>(ncorr_col(r));
        std::vector<int> corr_type;
        if (corr_type_col_ptr) {
          casacore::Vector<casacore::Int> vec = (*corr_type_col_ptr)(r);
          corr_type.resize(vec.size());
          for (size_t k = 0; k < vec.size(); k++)
            corr_type[k] = static_cast<int>(vec(k));
        }
        out.metadata().add_polarization(
            Polarization(pol_id, num_corr, std::move(corr_type)));
      }
    }

    for (size_t i = 0; i < dd_tab.nrow(); i++) {
      int dd_id = static_cast<int>(dd_id_col(i));
      int spw_id = static_cast<int>(dd_spw_col(i));
      int pol_id = static_cast<int>(dd_pol_col(i));
      const auto& spw = out.metadata().spectral_window(spw_id);
      int nchan = spw.nchan();
      int npol = 1;
      const Polarization* pol = out.metadata().find_polarization(pol_id);
      if (pol) npol = pol->num_corr();
      out.metadata().add_data_description(
          DataDescription(dd_id, spw_id, pol_id, nchan, npol));
    }
    out.metadata().build_index();

    read_antennas(dir, main_tab, out);

    casacore::ROArrayColumn<double> ref_col(field_tab, "REFERENCE_DIR");
    casacore::ROArrayColumn<double> phs_col(field_tab, "PHASE_DIR");
    casacore::ROScalarColumn<casacore::Int64> field_id_col(field_tab, "ID");
    for (size_t f = 0; f < field_tab.nrow(); f++) {
      FieldMetadata meta;
      meta.field_id = static_cast<int>(field_id_col(f));
      casacore::Vector<double> ref = ref_col(f);
      casacore::Vector<double> phs = phs_col(f);
      meta.reference_dir[0] = ref.size() > 0 ? ref(0) : 0.0;
      meta.reference_dir[1] = ref.size() > 1 ? ref(1) : 0.0;
      meta.phase_dir[0] = phs.size() > 0 ? phs(0) : 0.0;
      meta.phase_dir[1] = phs.size() > 1 ? phs(1) : 0.0;
      out.add_field(meta);
    }
  }

  void read_visibilities(casacore::Table& main_tab,
                         const std::string& dir,
                         const std::string& data_column,
                         const MSReadOptions& options,
                         MeasurementSet& out) {
    bool want_model =
        options.read_model && table_has_column(main_tab, "MODEL_DATA");
    bool has_weight_spectrum = table_has_column(main_tab, "WEIGHT_SPECTRUM");
    std::string query =
        "select ANTENNA1,ANTENNA2,FIELD_ID,TIME,DATA_DESC_ID,UVW,WEIGHT," +
        data_column + ",FLAG";
    if (has_weight_spectrum) query += ",WEIGHT_SPECTRUM";
    if (want_model) query += ",MODEL_DATA";
    query += " from " + dir + " where !FLAG_ROW and ANY(!FLAG)";
    query += " ORDERBY FIELD_ID, ANTENNA1, ANTENNA2, TIME, DATA_DESC_ID";
    if (options.order_by_w) query += ", UVW[2]";
    if (options.random_probability < 1.0f)
      query += " and RAND()<" + std::to_string(options.random_probability);

    casacore::Table query_tab = casacore::tableCommand(query.c_str());
    if (query_tab.nrow() == 0) return;

    casacore::ROScalarColumn<casacore::Int> ant1_col(query_tab, "ANTENNA1");
    casacore::ROScalarColumn<casacore::Int> ant2_col(query_tab, "ANTENNA2");
    casacore::ROScalarColumn<casacore::Int> field_id_col(query_tab, "FIELD_ID");
    casacore::ROScalarColumn<double> time_col(query_tab, "TIME");
    casacore::ROScalarColumn<casacore::Int> dd_id_col(query_tab, "DATA_DESC_ID");
    casacore::ROArrayColumn<double> uvw_col(query_tab, "UVW");
    casacore::ROArrayColumn<float> weight_col(query_tab, "WEIGHT");
    casacore::ROArrayColumn<casacore::Complex> data_col(query_tab, data_column);
    casacore::ROArrayColumn<bool> flag_col(query_tab, "FLAG");

    bool read_weight_spectrum =
        has_weight_spectrum && query_tab.tableDesc().isColumn("WEIGHT_SPECTRUM");
    std::unique_ptr<casacore::ROArrayColumn<float>> weight_spectrum_col_ptr;
    if (read_weight_spectrum)
      weight_spectrum_col_ptr =
          std::make_unique<casacore::ROArrayColumn<float>>(query_tab,
                                                           "WEIGHT_SPECTRUM");

    bool read_model = want_model && query_tab.tableDesc().isColumn("MODEL_DATA");
    std::unique_ptr<casacore::ROArrayColumn<casacore::Complex>> model_col_ptr;
    if (read_model)
      model_col_ptr =
          std::make_unique<casacore::ROArrayColumn<casacore::Complex>>(
              query_tab, "MODEL_DATA");

    for (size_t k = 0; k < query_tab.nrow(); k++) {
      int antenna1 = ant1_col(k);
      int antenna2 = ant2_col(k);
      int field_id = field_id_col(k);
      double time = time_col(k);
      int data_desc_id = dd_id_col(k);

      if (field_id < 0 || static_cast<size_t>(field_id) >= out.num_fields())
        continue;

      casacore::Vector<double> uvw = uvw_col(k);
      casacore::Vector<float> weight = weight_col(k);
      casacore::Matrix<casacore::Complex> dataMat = data_col(k);
      casacore::Matrix<bool> flag = flag_col(k);

      int npol = dataMat.nrow();
      int nchan = dataMat.ncolumn();

      const DataDescription* dd = out.metadata().find_data_description(data_desc_id);
      const std::vector<int>* selected_pol = nullptr;
      if (dd && dd->has_pol_selection())
        selected_pol = &dd->selected_pol_indices();

      TimeSample ts(data_desc_id, time);
      ts.set_uvw(make_double3(uvw(0), uvw(1), uvw(2)));
      if (selected_pol) {
        std::vector<float> sel_weight;
        sel_weight.reserve(selected_pol->size());
        for (int p : *selected_pol)
          sel_weight.push_back(p < static_cast<int>(weight.size()) ? weight(p) : 0.f);
        ts.set_weight(std::move(sel_weight));
      } else {
        ts.set_weight(std::vector<float>(weight.begin(), weight.end()));
      }
      ts.set_sigma(std::vector<float>());

      casacore::Matrix<float> weightSpectrum;
      if (read_weight_spectrum && weight_spectrum_col_ptr)
        weightSpectrum = (*weight_spectrum_col_ptr)(k);

      for (int j = 0; j < nchan; j++) {
        if (selected_pol) {
          for (size_t sp = 0; sp < selected_pol->size(); sp++) {
            int p = (*selected_pol)[sp];
            if (p < 0 || p >= npol) continue;
            float w = weight(p);
            if (read_weight_spectrum && weight_spectrum_col_ptr &&
                weightSpectrum.nrow() > 0 && weightSpectrum.ncolumn() > 0)
              w = weightSpectrum(p, j);
            if (w <= 0.0f || flag(p, j)) continue;
            cufftComplex Vo = make_cuFloatComplex(dataMat(p, j).real(), dataMat(p, j).imag());
            cufftComplex Vm = {0.f, 0.f};
            if (read_model && model_col_ptr) {
              casacore::Matrix<casacore::Complex> modelMat = (*model_col_ptr)(k);
              Vm = make_cuFloatComplex(modelMat(p, j).real(), modelMat(p, j).imag());
            }
            cufftComplex Vr = make_cuFloatComplex(Vo.x - Vm.x, Vo.y - Vm.y);
            ts.add_visibility(j, static_cast<int>(sp), Vo, Vm, Vr, w, false);
          }
        } else {
          for (int p = 0; p < npol; p++) {
            float w = weight(p);
            if (read_weight_spectrum && weight_spectrum_col_ptr &&
                weightSpectrum.nrow() > 0 && weightSpectrum.ncolumn() > 0)
              w = weightSpectrum(p, j);
            if (w <= 0.0f || flag(p, j)) continue;
            cufftComplex Vo = make_cuFloatComplex(dataMat(p, j).real(), dataMat(p, j).imag());
            cufftComplex Vm = {0.f, 0.f};
            if (read_model && model_col_ptr) {
              casacore::Matrix<casacore::Complex> modelMat = (*model_col_ptr)(k);
              Vm = make_cuFloatComplex(modelMat(p, j).real(), modelMat(p, j).imag());
            }
            cufftComplex Vr = make_cuFloatComplex(Vo.x - Vm.x, Vo.y - Vm.y);
            ts.add_visibility(j, p, Vo, Vm, Vr, w, false);
          }
        }
      }

      Field& field = out.field(static_cast<size_t>(field_id));
      Baseline& bl = field.baseline(antenna1, antenna2);
      bl.add_time_sample(std::move(ts));
    }

    for (size_t f = 0; f < out.num_fields(); f++) {
      for (auto& bl : out.field(f).baselines())
        bl.sort_by_time();
    }
  }
};

std::unique_ptr<MSReader> create_ms_reader() {
  return std::make_unique<CasacoreMSReader>();
}

}  // namespace ms
}  // namespace gpuvmem
