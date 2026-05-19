#ifndef GPUVMEM_MS_BASELINE_H
#define GPUVMEM_MS_BASELINE_H

#include "ms/time_sample.h"

#include <algorithm>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * One baseline (antenna1, antenna2). Holds all time samples for this baseline.
 * Decouples from dataset layout; aligned with image synthesis (per baseline).
 */
class Baseline {
 public:
  Baseline() = default;
  Baseline(int antenna1, int antenna2)
      : antenna1_(antenna1), antenna2_(antenna2) {}

  int antenna1() const { return antenna1_; }
  int antenna2() const { return antenna2_; }
  void set_antenna1(int a) { antenna1_ = a; }
  void set_antenna2(int a) { antenna2_ = a; }

  std::vector<TimeSample>& time_samples() { return time_samples_; }
  const std::vector<TimeSample>& time_samples() const { return time_samples_; }
  void add_time_sample(TimeSample ts) { time_samples_.push_back(std::move(ts)); }

  void sort_by_time() {
    std::sort(time_samples_.begin(), time_samples_.end(),
              [](const TimeSample& a, const TimeSample& b) {
                return a.time() < b.time();
              });
  }

 private:
  int antenna1_{0};
  int antenna2_{0};
  std::vector<TimeSample> time_samples_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_BASELINE_H
