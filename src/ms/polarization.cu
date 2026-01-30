#include "ms/polarization.h"
#include "ms/measurement_set.h"
#include "ms/time_sample.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace gpuvmem {
namespace ms {

namespace {

constexpr float half = 0.5f;
constexpr std::complex<float> i_(0.f, 1.f);

/** Canonical correlation order for circular: RR, LL, RL, LR. */
constexpr int kCircularOrder[4] = {CorrType::RR, CorrType::LL, CorrType::RL,
                                  CorrType::LR};
/** Canonical correlation order for linear: XX, YY, XY, YX. */
constexpr int kLinearOrder[4] = {CorrType::XX, CorrType::YY, CorrType::XY,
                                 CorrType::YX};

int canonical_index_circular(int corr_type) {
  for (int i = 0; i < 4; ++i)
    if (kCircularOrder[i] == corr_type) return i;
  return -1;
}

int canonical_index_linear(int corr_type) {
  for (int i = 0; i < 4; ++i)
    if (kLinearOrder[i] == corr_type) return i;
  return -1;
}

/** Stokes name to index: I=0, Q=1, U=2, V=3. */
int stokes_index(const std::string& s) {
  if (s == "I") return 0;
  if (s == "Q") return 1;
  if (s == "U") return 2;
  if (s == "V") return 3;
  return -1;
}

}  // namespace

std::string correlation_name(int corr_type) {
  switch (corr_type) {
    case CorrType::RR:
      return "RR";
    case CorrType::LL:
      return "LL";
    case CorrType::RL:
      return "RL";
    case CorrType::LR:
      return "LR";
    case CorrType::XX:
      return "XX";
    case CorrType::YY:
      return "YY";
    case CorrType::XY:
      return "XY";
    case CorrType::YX:
      return "YX";
    default:
      return "";
  }
}

bool is_circular(int corr_type) {
  return corr_type == CorrType::RR || corr_type == CorrType::LL ||
         corr_type == CorrType::RL || corr_type == CorrType::LR;
}

bool is_linear(int corr_type) {
  return corr_type == CorrType::XX || corr_type == CorrType::YY ||
         corr_type == CorrType::XY || corr_type == CorrType::YX;
}

FeedType PolarizationHelper::feed_type(int pol_id) const {
  if (!metadata_) return FeedType::Mixed;
  const Polarization* pol = metadata_->find_polarization(pol_id);
  if (!pol) return FeedType::Mixed;
  const std::vector<int>& ct = pol->corr_type();
  if (ct.empty()) return FeedType::Mixed;
  bool has_circ = false;
  bool has_lin = false;
  for (int c : ct) {
    if (is_circular(c)) has_circ = true;
    if (is_linear(c)) has_lin = true;
  }
  if (has_circ && has_lin) return FeedType::Mixed;
  if (has_circ) return FeedType::Circular;
  if (has_lin) return FeedType::Linear;
  return FeedType::Mixed;
}

FeedType PolarizationHelper::feed_type_for_data_desc(int data_desc_id) const {
  if (!metadata_) return FeedType::Mixed;
  const DataDescription* dd = metadata_->find_data_description(data_desc_id);
  if (!dd) return FeedType::Mixed;
  return feed_type(dd->polarization_id());
}

std::vector<std::string> PolarizationHelper::correlation_names(int pol_id) const {
  std::vector<std::string> out;
  if (!metadata_) return out;
  const Polarization* pol = metadata_->find_polarization(pol_id);
  if (!pol) return out;
  for (int c : pol->corr_type()) out.push_back(correlation_name(c));
  return out;
}

std::vector<std::string> PolarizationHelper::available_stokes(int pol_id) const {
  std::vector<std::string> out;
  if (!metadata_) return out;
  const Polarization* pol = metadata_->find_polarization(pol_id);
  if (!pol) return out;
  const std::vector<int>& ct = pol->corr_type();
  FeedType ft = feed_type(pol_id);

  if (ft == FeedType::Circular) {
    bool has_rr = false, has_ll = false, has_rl = false, has_lr = false;
    for (int c : ct) {
      if (c == CorrType::RR) has_rr = true;
      if (c == CorrType::LL) has_ll = true;
      if (c == CorrType::RL) has_rl = true;
      if (c == CorrType::LR) has_lr = true;
    }
    if (has_rr && has_ll) {
      out.push_back("I");
      out.push_back("V");
    }
    if (has_rl && has_lr) {
      out.push_back("Q");
      out.push_back("U");
    }
    // Return in standard order I,Q,U,V when all four present
    if (out.size() == 4u)
      out = std::vector<std::string>{"I", "Q", "U", "V"};
  } else if (ft == FeedType::Linear) {
    bool has_xx = false, has_yy = false, has_xy = false, has_yx = false;
    for (int c : ct) {
      if (c == CorrType::XX) has_xx = true;
      if (c == CorrType::YY) has_yy = true;
      if (c == CorrType::XY) has_xy = true;
      if (c == CorrType::YX) has_yx = true;
    }
    if (has_xx && has_yy) {
      out.push_back("I");
      out.push_back("Q");
    }
    if (has_xy && has_yx) {
      out.push_back("U");
      out.push_back("V");
    }
  }
  return out;
}

std::vector<std::string> PolarizationHelper::available_stokes_for_data_desc(
    int data_desc_id) const {
  if (!metadata_) return {};
  const DataDescription* dd = metadata_->find_data_description(data_desc_id);
  if (!dd) return {};
  return available_stokes(dd->polarization_id());
}

PolarizationHelper::Matrix PolarizationHelper::build_corr_to_stokes_circular(
    const std::vector<int>& /*corr_type*/) {
  // Rows: I, Q, U, V. Cols: RR, LL, RL, LR (canonical).
  // I = 0.5*RR + 0.5*LL, V = 0.5*RR - 0.5*LL
  // Q = 0.5*RL + 0.5*LR, U = -i/2*RL + i/2*LR
  Matrix m(4, std::vector<Complex>(4, 0.f));
  m[0][0] = half;   // I from RR
  m[0][1] = half;   // I from LL
  m[1][2] = half;   // Q from RL
  m[1][3] = half;   // Q from LR
  m[2][2] = -i_ * half;  // U from RL
  m[2][3] = i_ * half;    // U from LR
  m[3][0] = half;   // V from RR
  m[3][1] = -half; // V from LL
  return m;
}

PolarizationHelper::Matrix PolarizationHelper::build_corr_to_stokes_linear(
    const std::vector<int>& /*corr_type*/) {
  // Rows: I, Q, U, V. Cols: XX, YY, XY, YX (canonical).
  // I = 0.5*XX + 0.5*YY, Q = 0.5*XX - 0.5*YY
  // U = 0.5*XY + 0.5*YX, V = -i/2*XY + i/2*YX
  Matrix m(4, std::vector<Complex>(4, 0.f));
  m[0][0] = half;   // I from XX
  m[0][1] = half;   // I from YY
  m[1][0] = half;   // Q from XX
  m[1][1] = -half;  // Q from YY
  m[2][2] = half;   // U from XY
  m[2][3] = half;   // U from YX
  m[3][2] = -i_ * half;  // V from XY
  m[3][3] = i_ * half;   // V from YX
  return m;
}

PolarizationHelper::Matrix PolarizationHelper::build_stokes_to_corr_circular(
    const std::vector<int>& /*corr_type*/) {
  // Rows: RR, LL, RL, LR. Cols: I, Q, U, V.
  // RR = I+V, LL = I-V, RL = Q+iU, LR = Q-iU
  Matrix m(4, std::vector<Complex>(4, 0.f));
  m[0][0] = 1.f;   // RR from I
  m[0][3] = 1.f;   // RR from V
  m[1][0] = 1.f;   // LL from I
  m[1][3] = -1.f;  // LL from V
  m[2][1] = 1.f;   // RL from Q
  m[2][2] = i_;    // RL from U
  m[3][1] = 1.f;   // LR from Q
  m[3][2] = -i_;   // LR from U
  return m;
}

PolarizationHelper::Matrix PolarizationHelper::build_stokes_to_corr_linear(
    const std::vector<int>& /*corr_type*/) {
  // Rows: XX, YY, XY, YX. Cols: I, Q, U, V.
  // XX = I+Q, YY = I-Q, XY = U+iV, YX = U-iV
  Matrix m(4, std::vector<Complex>(4, 0.f));
  m[0][0] = 1.f;   // XX from I
  m[0][1] = 1.f;   // XX from Q
  m[1][0] = 1.f;   // YY from I
  m[1][1] = -1.f;  // YY from Q
  m[2][2] = 1.f;   // XY from U
  m[2][3] = i_;    // XY from V
  m[3][2] = 1.f;   // YX from U
  m[3][3] = -i_;   // YX from V
  return m;
}

PolarizationHelper::Matrix PolarizationHelper::select_corr_to_stokes(
    const Matrix& full_c2s, const std::vector<int>& corr_type,
    const std::vector<std::string>& stokes) {
  const bool is_circ = corr_type.size() > 0 && is_circular(corr_type[0]);
  Matrix out;
  out.reserve(stokes.size());
  for (const std::string& s : stokes) {
    int si = stokes_index(s);
    if (si < 0) continue;
    std::vector<Complex> row(corr_type.size(), 0.f);
    for (size_t c = 0; c < corr_type.size(); ++c) {
      int can_idx = is_circ ? canonical_index_circular(corr_type[c])
                            : canonical_index_linear(corr_type[c]);
      if (can_idx >= 0 && si < 4)
        row[c] = full_c2s[si][can_idx];
    }
    out.push_back(std::move(row));
  }
  return out;
}

PolarizationHelper::Matrix PolarizationHelper::select_stokes_to_corr(
    const Matrix& full_s2c, const std::vector<int>& corr_type,
    const std::vector<std::string>& stokes) {
  const bool is_circ = corr_type.size() > 0 && is_circular(corr_type[0]);
  Matrix out(corr_type.size(),
             std::vector<Complex>(stokes.size(), 0.f));
  for (size_t r = 0; r < corr_type.size(); ++r) {
    int can_idx = is_circ ? canonical_index_circular(corr_type[r])
                          : canonical_index_linear(corr_type[r]);
    if (can_idx < 0) continue;
    for (size_t c = 0; c < stokes.size(); ++c) {
      int si = stokes_index(stokes[c]);
      if (si >= 0 && can_idx < 4)
        out[r][c] = full_s2c[can_idx][si];
    }
  }
  return out;
}

PolarizationHelper::Matrix PolarizationHelper::get_correlation_to_stokes_matrix(
    int pol_id, const std::vector<std::string>& stokes_list) const {
  if (!metadata_) return {};
  const Polarization* pol = metadata_->find_polarization(pol_id);
  if (!pol) return {};
  const std::vector<int>& ct = pol->corr_type();
  if (ct.empty()) return {};
  FeedType ft = feed_type(pol_id);
  Matrix full =
      (ft == FeedType::Circular) ? build_corr_to_stokes_circular(ct)
                                 : build_corr_to_stokes_linear(ct);
  return select_corr_to_stokes(full, ct, stokes_list);
}

PolarizationHelper::Matrix
PolarizationHelper::get_correlation_to_stokes_matrix_for_data_desc(
    int data_desc_id, const std::vector<std::string>& stokes_list) const {
  if (!metadata_) return {};
  const DataDescription* dd = metadata_->find_data_description(data_desc_id);
  if (!dd) return {};
  return get_correlation_to_stokes_matrix(dd->polarization_id(), stokes_list);
}

PolarizationHelper::Matrix PolarizationHelper::get_stokes_to_correlation_matrix(
    int pol_id, const std::vector<std::string>& stokes_list) const {
  if (!metadata_) return {};
  const Polarization* pol = metadata_->find_polarization(pol_id);
  if (!pol) return {};
  const std::vector<int>& ct = pol->corr_type();
  if (ct.empty()) return {};
  FeedType ft = feed_type(pol_id);
  Matrix full =
      (ft == FeedType::Circular) ? build_stokes_to_corr_circular(ct)
                                 : build_stokes_to_corr_linear(ct);
  return select_stokes_to_corr(full, ct, stokes_list);
}

PolarizationHelper::Matrix
PolarizationHelper::get_stokes_to_correlation_matrix_for_data_desc(
    int data_desc_id, const std::vector<std::string>& stokes_list) const {
  if (!metadata_) return {};
  const DataDescription* dd = metadata_->find_data_description(data_desc_id);
  if (!dd) return {};
  return get_stokes_to_correlation_matrix(dd->polarization_id(), stokes_list);
}

static cufftComplex to_cufft(std::complex<float> z) {
  return make_cuFloatComplex(z.real(), z.imag());
}
static std::complex<float> from_cufft(cufftComplex v) {
  return std::complex<float>(v.x, v.y);
}

/** sigma² = 1/w; return 0 if w <= 0 to avoid inf. */
static float variance_from_weight(float w) {
  if (w <= 0.f || std::isnan(w)) return 0.f;
  return 1.f / w;
}

bool stokes_supported_by_metadata(const MeasurementSetMetadata& meta,
                                  const std::vector<std::string>& requested_stokes) {
  if (requested_stokes.empty()) return true;
  PolarizationHelper helper(&meta);
  for (const DataDescription& dd : meta.data_descriptions()) {
    std::vector<std::string> available =
        helper.available_stokes_for_data_desc(dd.data_desc_id());
    for (const std::string& req : requested_stokes) {
      bool found = false;
      for (const std::string& a : available) {
        if (a == req) {
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
  }
  return true;
}

bool correlations_to_stokes(MeasurementSet& ms,
                            const std::vector<std::string>& stokes_list) {
  if (stokes_list.empty()) return false;
  PolarizationHelper helper(&ms.metadata());
  const size_t num_stokes = stokes_list.size();

  for (size_t f = 0; f < ms.num_fields(); f++) {
    Field& field = ms.field(f);
    for (Baseline& bl : field.baselines()) {
      for (TimeSample& ts : bl.time_samples()) {
        int data_desc_id = ts.data_desc_id();
        DataDescription* dd = ms.metadata().find_data_description(data_desc_id);
        if (!dd) continue;
        int npol = dd->npol();
        if (npol <= 0) continue;
        const std::vector<int>* sel = dd->has_pol_selection()
                                        ? &dd->selected_pol_indices()
                                        : nullptr;

        PolarizationHelper::Matrix M =
            helper.get_correlation_to_stokes_matrix_for_data_desc(
                data_desc_id, stokes_list);
        if (M.empty() || M[0].size() < static_cast<size_t>(npol)) continue;

        std::vector<TimeSample::VisSample>& vis = ts.visibilities();
        if (vis.empty()) continue;
        int nchan = static_cast<int>(vis.size()) / npol;
        if (nchan * npol != static_cast<int>(vis.size())) continue;

        std::vector<TimeSample::VisSample> new_vis;
        new_vis.reserve(nchan * num_stokes);
        std::vector<float> new_weights(num_stokes, 0.f);
        std::vector<int> new_weight_count(num_stokes, 0);

        for (int j = 0; j < nchan; j++) {
          std::vector<std::complex<float>> Vo_in(npol), Vm_in(npol),
              Vr_in(npol);
          std::vector<float> w_in(npol);
          for (int p = 0; p < npol; p++) {
            size_t idx = j * npol + p;
            Vo_in[p] = from_cufft(vis[idx].Vo);
            Vm_in[p] = from_cufft(vis[idx].Vm);
            Vr_in[p] = from_cufft(vis[idx].Vr);
            w_in[p] = vis[idx].imaging_weight;
          }
          for (size_t s = 0; s < num_stokes; s++) {
            std::complex<float> Vo_s(0.f, 0.f), Vm_s(0.f, 0.f), Vr_s(0.f, 0.f);
            float sigma_sq = 0.f;
            for (int p = 0; p < npol; p++) {
              int col = sel ? (*sel)[p] : p;
              if (col >= static_cast<int>(M[s].size())) continue;
              PolarizationHelper::Complex a = M[s][col];
              Vo_s += a * Vo_in[p];
              Vm_s += a * Vm_in[p];
              Vr_s += a * Vr_in[p];
              float sig_p_sq = variance_from_weight(w_in[p]);
              sigma_sq += (a.real() * a.real() + a.imag() * a.imag()) * sig_p_sq;
            }
            float w_s = (sigma_sq > 0.f && !std::isnan(sigma_sq))
                            ? (1.f / sigma_sq)
                            : 0.f;
            new_vis.push_back(
                {j, static_cast<int>(s), to_cufft(Vo_s), to_cufft(Vm_s),
                 to_cufft(Vr_s), false, w_s, w_s});
            if (w_s > 0.f) {
              new_weights[s] += w_s;
              new_weight_count[s]++;
            }
          }
        }
        for (size_t s = 0; s < num_stokes; s++)
          if (new_weight_count[s] > 0)
            new_weights[s] /= new_weight_count[s];
          else
            new_weights[s] = 1.f;

        vis = std::move(new_vis);
        ts.set_weight(std::move(new_weights));
        dd->set_npol(static_cast<int>(num_stokes));
        dd->set_selected_pol_indices({});
      }
    }
  }
  ms.set_storage_mode(StorageMode::Stokes);
  ms.set_stored_stokes_list(stokes_list);
  return true;
}

bool stokes_to_correlations(MeasurementSet& ms) {
  if (ms.storage_mode() != StorageMode::Stokes) return true;
  const std::vector<std::string>& stokes_list = ms.stored_stokes_list();
  if (stokes_list.empty()) return false;
  PolarizationHelper helper(&ms.metadata());
  const size_t num_stokes = stokes_list.size();

  for (size_t f = 0; f < ms.num_fields(); f++) {
    Field& field = ms.field(f);
    for (Baseline& bl : field.baselines()) {
      for (TimeSample& ts : bl.time_samples()) {
        int data_desc_id = ts.data_desc_id();
        DataDescription* dd = ms.metadata().find_data_description(data_desc_id);
        if (!dd) continue;
        int pol_id = dd->polarization_id();
        const Polarization* pol = ms.metadata().find_polarization(pol_id);
        if (!pol) continue;
        int num_corr = pol->num_corr();
        if (num_corr <= 0) continue;

        PolarizationHelper::Matrix M =
            helper.get_stokes_to_correlation_matrix(pol_id, stokes_list);
        if (M.size() < static_cast<size_t>(num_corr) || M[0].size() < num_stokes)
          continue;

        std::vector<TimeSample::VisSample>& vis = ts.visibilities();
        if (vis.empty()) continue;
        int nchan = static_cast<int>(vis.size()) / static_cast<int>(num_stokes);
        if (nchan * static_cast<int>(num_stokes) != static_cast<int>(vis.size()))
          continue;

        std::vector<TimeSample::VisSample> new_vis;
        new_vis.reserve(nchan * num_corr);
        std::vector<float> new_weights(num_corr, 0.f);
        std::vector<int> new_weight_count(num_corr, 0);

        for (int j = 0; j < nchan; j++) {
          std::vector<std::complex<float>> Vo_in(num_stokes), Vm_in(num_stokes),
              Vr_in(num_stokes);
          std::vector<float> w_in(num_stokes);
          for (size_t s = 0; s < num_stokes; s++) {
            size_t idx = j * num_stokes + s;
            Vo_in[s] = from_cufft(vis[idx].Vo);
            Vm_in[s] = from_cufft(vis[idx].Vm);
            Vr_in[s] = from_cufft(vis[idx].Vr);
            w_in[s] = vis[idx].imaging_weight;
          }
          for (int c = 0; c < num_corr; c++) {
            std::complex<float> Vo_c(0.f, 0.f), Vm_c(0.f, 0.f), Vr_c(0.f, 0.f);
            float sigma_sq = 0.f;
            for (size_t s = 0; s < num_stokes; s++) {
              PolarizationHelper::Complex a = M[c][s];
              Vo_c += a * Vo_in[s];
              Vm_c += a * Vm_in[s];
              Vr_c += a * Vr_in[s];
              float sig_s_sq = variance_from_weight(w_in[s]);
              sigma_sq += (a.real() * a.real() + a.imag() * a.imag()) * sig_s_sq;
            }
            float w_c = (sigma_sq > 0.f && !std::isnan(sigma_sq))
                           ? (1.f / sigma_sq)
                           : 0.f;
            new_vis.push_back(
                {j, c, to_cufft(Vo_c), to_cufft(Vm_c), to_cufft(Vr_c), false,
                 w_c, w_c});
            if (w_c > 0.f) {
              new_weights[c] += w_c;
              new_weight_count[c]++;
            }
          }
        }
        for (int c = 0; c < num_corr; c++)
          if (new_weight_count[c] > 0)
            new_weights[c] /= new_weight_count[c];
          else
            new_weights[c] = 1.f;

        vis = std::move(new_vis);
        ts.set_weight(std::move(new_weights));
        dd->set_npol(num_corr);
        dd->set_selected_pol_indices({});
      }
    }
  }
  ms.set_storage_mode(StorageMode::Correlations);
  ms.set_stored_stokes_list({});
  return true;
}

}  // namespace ms
}  // namespace gpuvmem
