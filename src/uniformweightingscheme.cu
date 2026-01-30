#include "uniformweightingscheme.cuh"
#include "functions.cuh"
#include "ms/measurement_set.h"
#include "ms/time_sample.h"

#include <cmath>
#include <iostream>
#include <map>
#include <vector>

UniformWeightingScheme::UniformWeightingScheme() : WeightingScheme(){};
UniformWeightingScheme::UniformWeightingScheme(int threads)
    : WeightingScheme(threads){};
UniformWeightingScheme::UniformWeightingScheme(int threads, UVTaper* uvtaper)
    : WeightingScheme(threads, uvtaper){};

namespace {
struct GroupKey {
  int field_id;
  int data_desc_id;
  int chan;
  int pol;
  bool operator<(const GroupKey& o) const {
    if (field_id != o.field_id) return field_id < o.field_id;
    if (data_desc_id != o.data_desc_id) return data_desc_id < o.data_desc_id;
    if (chan != o.chan) return chan < o.chan;
    return pol < o.pol;
  }
};
struct VisEntry {
  gpuvmem::ms::TimeSample::VisSample* vis;
  double ux, uy;
  float w;
};
}  // namespace

void UniformWeightingScheme::apply(std::vector<gpuvmem::ms::MSWithGPU>& d) {
  std::cout << "Running Uniform weighting scheme with " << this->threads
            << " threads" << std::endl;

  std::map<GroupKey, std::vector<VisEntry>> groups;
  for (auto& dw : d) {
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    for (size_t f = 0; f < dw.ms.num_fields(); f++) {
      gpuvmem::ms::Field& field = dw.ms.field(f);
      int field_id = field.field_id();
      for (auto& bl : field.baselines()) {
        for (auto& ts : bl.time_samples()) {
          int ddid = ts.data_desc_id();
          const gpuvmem::ms::DataDescription* dd =
              meta.find_data_description(ddid);
          if (!dd) continue;
          const gpuvmem::ms::SpectralWindow* spw =
              meta.find_spectral_window(dd->spectral_window_id());
          if (!spw) continue;

          double3 uvw = ts.uvw();
          for (auto& vis : ts.visibilities()) {
            float freq = static_cast<float>(spw->frequency(vis.chan));
            double ux = metres_to_lambda(uvw.x, freq);
            double uy = metres_to_lambda(uvw.y, freq);
            if (ux < 0.0) {
              ux = -ux;
              uy = -uy;
            }
            GroupKey key{field_id, ddid, vis.chan, vis.pol};
            groups[key].push_back({&vis, ux, uy, vis.imaging_weight});
          }
        }
      }
    }
  }

  std::vector<float> g_weights(M * N);
  for (auto& kv : groups) {
    std::fill(g_weights.begin(), g_weights.end(), 0.0f);
    for (const auto& e : kv.second) {
      double grid_pos_x = e.ux / fabs(deltau);
      double grid_pos_y = e.uy / fabs(deltav);
      int x = static_cast<int>(grid_pos_x + floor(N / 2.0) + 0.5);
      int y = static_cast<int>(grid_pos_y + floor(M / 2.0) + 0.5);
      if (x >= 0 && y >= 0 && x < N && y < M)
        g_weights[N * y + x] += e.w;
    }
#pragma omp parallel for schedule(static, 1) num_threads(this->threads)
    for (size_t z = 0; z < kv.second.size(); z++) {
      auto& e = kv.second[z];
      double grid_pos_x = e.ux / fabs(deltau);
      double grid_pos_y = e.uy / fabs(deltav);
      int x = static_cast<int>(grid_pos_x + floor(N / 2.0) + 0.5);
      int y = static_cast<int>(grid_pos_y + floor(M / 2.0) + 0.5);
      if (x >= 0 && y >= 0 && x < N && y < M)
        e.vis->imaging_weight /= g_weights[N * y + x];
      else
        e.vis->imaging_weight = 0.0f;
      if (this->uvtaper != NULL)
        e.vis->imaging_weight *= this->uvtaper->getValue(e.ux, e.uy);
    }
  }
}

namespace {
WeightingScheme* CreateWeightingScheme() {
  return new UniformWeightingScheme;
}
const std::string name = "Uniform";
const bool RegisteredUniformWeighting =
    registerCreationFunction<WeightingScheme, std::string>(
        name,
        CreateWeightingScheme);
};  // namespace
