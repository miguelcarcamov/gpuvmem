#ifndef GPUVMEM_MS_FIELD_H
#define GPUVMEM_MS_FIELD_H

#include "ms/baseline.h"
#include "ms/metadata.h"

#include <set>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * One field (FIELD_ID). Holds field metadata (phase center, reference center, etc.)
 * and visibility data (baselines). Does NOT own spectral windows; those live at MS level.
 */
class Field {
 public:
  Field() = default;
  explicit Field(const FieldMetadata& meta) : meta_(meta) {}

  const FieldMetadata& metadata() const { return meta_; }
  FieldMetadata& metadata() { return meta_; }

  int field_id() const { return meta_.field_id; }
  const std::string& name() const { return meta_.name; }
  const std::array<double, 2>& reference_dir() const {
    return meta_.reference_dir;
  }
  const std::array<double, 2>& phase_dir() const { return meta_.phase_dir; }
  void set_reference_dir(double ra_rad, double dec_rad) {
    meta_.reference_dir[0] = ra_rad;
    meta_.reference_dir[1] = dec_rad;
  }
  void set_phase_dir(double ra_rad, double dec_rad) {
    meta_.phase_dir[0] = ra_rad;
    meta_.phase_dir[1] = dec_rad;
  }

  Baseline& baseline(int antenna1, int antenna2) {
    Baseline* b = find_baseline(antenna1, antenna2);
    if (b) return *b;
    baselines_.emplace_back(antenna1, antenna2);
    return baselines_.back();
  }
  const Baseline* find_baseline(int antenna1, int antenna2) const {
    for (const auto& b : baselines_) {
      if (b.antenna1() == antenna1 && b.antenna2() == antenna2) return &b;
    }
    return nullptr;
  }
  Baseline* find_baseline(int antenna1, int antenna2) {
    for (auto& b : baselines_) {
      if (b.antenna1() == antenna1 && b.antenna2() == antenna2) return &b;
    }
    return nullptr;
  }
  const std::vector<Baseline>& baselines() const { return baselines_; }
  std::vector<Baseline>& baselines() { return baselines_; }

  std::vector<double> unique_times() const {
    std::set<double> times;
    for (const auto& bl : baselines_) {
      for (const auto& ts : bl.time_samples()) {
        times.insert(ts.time());
      }
    }
    return std::vector<double>(times.begin(), times.end());
  }

 private:
  FieldMetadata meta_;
  std::vector<Baseline> baselines_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_FIELD_H
