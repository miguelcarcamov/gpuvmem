#ifndef GPUVMEM_MS_GRIDDED_VISIBILITIES_H
#define GPUVMEM_MS_GRIDDED_VISIBILITIES_H

#include "ms/metadata.h"
#include "ms/measurement_set_metadata.h"

#include <cufft.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * Gridded visibility representation: same metadata as the dataset but
 * visibilities stored on a regular (u,v) grid for FFT-based synthesis.
 * Can be filled by gridding the MeasurementSet and kept in this structure.
 */
struct GriddedVisibilities {
  /** Grid size in u (number of cells). */
  int nu{0};
  /** Grid size in v (number of cells). */
  int nv{0};
  /** Cell size in u (wavelengths). */
  double du{0.0};
  /** Cell size in v (wavelengths). */
  double dv{0.0};
  /** Reference u (grid center), wavelengths. */
  double u0{0.0};
  /** Reference v (grid center), wavelengths. */
  double v0{0.0};

  /** Data description id this grid belongs to (frequency/pol context). */
  int data_desc_id{0};
  /** Field id. */
  int field_id{0};

  /**
   * Gridded complex visibility: host layout [nv][nu] or flat size nu*nv.
   * Owned; can be replaced by device buffer for GPU synthesis.
   */
  std::vector<cufftComplex> grid;
  /** Weight sum per cell (for natural weighting); same shape as grid. */
  std::vector<float> weight_sum;

  size_t size() const { return static_cast<size_t>(nu) * static_cast<size_t>(nv); }
  bool empty() const { return nu == 0 || nv == 0; }
  void clear() {
    nu = nv = 0;
    du = dv = u0 = v0 = 0.0;
    grid.clear();
    weight_sum.clear();
  }
};

/**
 * Container for gridded visibilities per (field, data_desc): same data
 * structure as the dataset, but gridded. Used after gridding step and
 * for FFT-based image synthesis.
 */
class GriddedVisibilitySet {
 public:
  GriddedVisibilitySet() = default;

  void set_metadata(std::shared_ptr<const MeasurementSetMetadata> meta) {
    metadata_ = std::move(meta);
  }
  const MeasurementSetMetadata* metadata() const { return metadata_.get(); }

  /** Add or replace gridded vis for (field_id, data_desc_id). */
  void set_grid(int field_id, int data_desc_id, GriddedVisibilities gv) {
    key_t k{field_id, data_desc_id};
    grids_[k] = std::move(gv);
  }
  const GriddedVisibilities* get_grid(int field_id, int data_desc_id) const {
    auto it = grids_.find({field_id, data_desc_id});
    return it == grids_.end() ? nullptr : &it->second;
  }
  GriddedVisibilities* get_grid(int field_id, int data_desc_id) {
    auto it = grids_.find({field_id, data_desc_id});
    return it == grids_.end() ? nullptr : &it->second;
  }

 private:
  using key_t = std::pair<int, int>;
  std::shared_ptr<const MeasurementSetMetadata> metadata_;
  std::map<key_t, GriddedVisibilities> grids_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_GRIDDED_VISIBILITIES_H
