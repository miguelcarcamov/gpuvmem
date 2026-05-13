#include "weighting/naturalweightingscheme.cuh"
#include "framework.cuh"
#include "utils/physics_utils.cuh"
#include "ms/measurement_set.h"
#include "ms/time_sample.h"

#include <iostream>

NaturalWeightingScheme::NaturalWeightingScheme() : WeightingScheme(){};
NaturalWeightingScheme::NaturalWeightingScheme(int threads)
    : WeightingScheme(threads){};
NaturalWeightingScheme::NaturalWeightingScheme(int threads, UVTaper* uvtaper)
    : WeightingScheme(threads, uvtaper){};

void NaturalWeightingScheme::configure(void* params) {
  if (params == nullptr) return;
  const float r = *static_cast<const float*>(params);
  if (r != 0.0f) {
    std::cerr << "WARNING: -R / robust parameter (" << r
              << ") has no effect with Natural weighting; use -W briggs (or robust) "
                 "to apply Briggs robustness.\n";
  }
}

void NaturalWeightingScheme::apply(std::vector<gpuvmem::ms::MSWithGPU>& d) {
  std::cout << "Running Natural weighting scheme with " << this->threads
            << " threads" << std::endl;

  for (auto& dw : d) {
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    for (size_t f = 0; f < dw.ms.num_fields(); f++) {
      gpuvmem::ms::Field& field = dw.ms.field(f);
      for (auto& bl : field.baselines()) {
        for (auto& ts : bl.time_samples()) {
          const gpuvmem::ms::DataDescription* dd =
              meta.find_data_description(ts.data_desc_id());
          if (!dd) continue;
          const gpuvmem::ms::SpectralWindow* spw =
              meta.find_spectral_window(dd->spectral_window_id());
          if (!spw) continue;

          double3 uvw = ts.uvw();
#pragma omp parallel for schedule(static, 1) num_threads(this->threads)
          for (size_t z = 0; z < ts.visibilities().size(); z++) {
            auto& vis = ts.visibilities()[z];
            float freq = static_cast<float>(spw->frequency(vis.chan));
            double ux = metres_to_lambda(uvw.x, freq);
            double uy = metres_to_lambda(uvw.y, freq);
            double uz = metres_to_lambda(uvw.z, freq);
            if (ux < 0.0) {
              ux = -ux;
              uy = -uy;
            }
            if (this->uvtaper != NULL)
              vis.imaging_weight *= this->uvtaper->getValue(ux, uy);
          }
        }
      }
    }
  }
}

namespace {
WeightingScheme* CreateWeightingScheme() {
  return new NaturalWeightingScheme;
}

const std::string name = "Natural";
const bool RegisteredNaturalWeighting =
    registerCreationFunction<WeightingScheme, std::string>(
        name,
        CreateWeightingScheme);
};  // namespace
