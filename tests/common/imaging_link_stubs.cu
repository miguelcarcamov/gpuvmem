// TU kept for weighting/CKernel factory targets: GPU count + optional default geometry init.
#include "imaging_geometry.hh"
#include "legacy_imaging_globals.hh"

int num_gpus = 1;
float* initial_values = nullptr;

namespace {
struct InitDefaults {
  InitDefaults() {
    gpuvmem::test::apply_legacy_imaging_globals(
        gpuvmem::test::dataset_geometry_presets().front());
  }
};
InitDefaults kInitDefaults;
}  // namespace
