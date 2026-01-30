#include "radialweightingscheme.cuh"
#include "functions.cuh"
#include "ms/measurement_set.h"
#include "ms/time_sample.h"

#include <iostream>

RadialWeightingScheme::RadialWeightingScheme() : WeightingScheme(){};
RadialWeightingScheme::RadialWeightingScheme(int threads)
    : WeightingScheme(threads){};

void RadialWeightingScheme::apply(std::vector<gpuvmem::ms::MSWithGPU>& d) {
  std::cout << "Running weighting scheme with " << this->threads << " threads"
            << std::endl;

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
            vis.imaging_weight *= distance(ux, uy, 0.0f, 0.0f);
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
  return new RadialWeightingScheme;
}

const std::string name = "Radial";
const bool RegisteredRadialWeighting =
    registerCreationFunction<WeightingScheme, std::string>(
        name,
        CreateWeightingScheme);
};  // namespace
