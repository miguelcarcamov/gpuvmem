#include "briggsweightingscheme.cuh"
#include "functions.cuh"
#include "ms/measurement_set.h"
#include "ms/time_sample.h"

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

BriggsWeightingScheme::BriggsWeightingScheme() : WeightingScheme(){};
BriggsWeightingScheme::BriggsWeightingScheme(int threads)
    : WeightingScheme(threads){};
BriggsWeightingScheme::BriggsWeightingScheme(int threads, UVTaper* uvtaper)
    : WeightingScheme(threads, uvtaper){};

float BriggsWeightingScheme::getRobustParam() {
  return this->robust_param;
};

void BriggsWeightingScheme::setRobustParam(float robust_param) {
  if (robust_param >= -2.0 && robust_param <= 2.0) {
    this->robust_param = robust_param;
  } else {
    std::cout << "Error. Robust parameter must have values between -2.0 and 2.0"
              << std::endl;
    exit(-1);
  }
};

void BriggsWeightingScheme::configure(void* params) {
  float robust_param = *((float*)params);
  this->setRobustParam(robust_param);
  std::cout << "Using robust " << this->getRobustParam()
            << " for Briggs weighting" << std::endl;
};

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

void BriggsWeightingScheme::apply(std::vector<gpuvmem::ms::MSWithGPU>& d) {
  std::cout << "Running Briggs weighting scheme with " << this->threads
            << " threads" << std::endl;

  std::map<GroupKey, std::vector<VisEntry>> groups;
  float sum_original_weights = 0.0f;

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
            sum_original_weights += vis.imaging_weight;
          }
        }
      }
    }
  }

  std::vector<float> g_weights(M * N, 0.0f);
  for (const auto& kv : groups) {
    for (const auto& e : kv.second) {
      double grid_pos_x = e.ux / fabs(deltau);
      double grid_pos_y = e.uy / fabs(deltav);
      int x = static_cast<int>(grid_pos_x + floor(N / 2.0) + 0.5);
      int y = static_cast<int>(grid_pos_y + floor(M / 2.0) + 0.5);
      if (x >= 0 && y >= 0 && x < N && y < M)
        g_weights[N * y + x] += e.w;
    }
  }
  float sum_gridded_weights_squared = 0.0f;
  for (long m = 0; m < M; m++)
    for (long n = N / 2; n < N; n++)
      sum_gridded_weights_squared += g_weights[N * m + n] * g_weights[N * m + n];

  float average_weights = sum_gridded_weights_squared / sum_original_weights;
  float f_squared =
      (5.0f * powf(10.0f, -this->getRobustParam())) *
      (5.0f * powf(10.0f, -this->getRobustParam())) / average_weights;

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
        e.vis->imaging_weight /= (1.0f + g_weights[N * y + x] * f_squared);
      else
        e.vis->imaging_weight = 0.0f;
      if (this->uvtaper != NULL)
        e.vis->imaging_weight *= this->uvtaper->getValue(e.ux, e.uy);
    }
  }
}

namespace {
WeightingScheme* CreateWeightingScheme() {
  return new BriggsWeightingScheme;
}

const std::string name = "Briggs";
const bool RegisteredBriggsWeighting =
    registerCreationFunction<WeightingScheme, std::string>(
        name,
        CreateWeightingScheme);
};  // namespace
