#ifndef GPUVMEM_MS_TIME_SAMPLE_H
#define GPUVMEM_MS_TIME_SAMPLE_H

#include <cufft.h>
#include <cuda_runtime.h>

#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * One time sample = one MAIN table row = one (baseline, time) for one DATA_DESC_ID.
 * Holds UVW once; per (chan, pol): weight (from MS), imaging_weight (used for imaging).
 *
 * Weight priority from MS: 1) WEIGHT_SPECTRUM if present (per chan, pol),
 * 2) WEIGHT broadcast to all channels (per pol). imaging_weight is set from that
 * at read time; weighting schemes modify imaging_weight only (not the raw weight).
 */
class TimeSample {
 public:
  struct VisSample {
    int chan{0};
    int pol{0};
    cufftComplex Vo{0.f, 0.f};
    cufftComplex Vm{0.f, 0.f};
    cufftComplex Vr{0.f, 0.f};
    bool flagged{false};
    /** From MS: WEIGHT_SPECTRUM(chan,pol) if present, else WEIGHT(pol) broadcast. */
    float weight{1.f};
    /** Used for imaging (chi2, gridding); weighting schemes modify this only. */
    float imaging_weight{1.f};
  };

  TimeSample() = default;
  explicit TimeSample(int data_desc_id, double time)
      : data_desc_id_(data_desc_id), time_(time) {}

  int data_desc_id() const { return data_desc_id_; }
  double time() const { return time_; }

  const double3& uvw() const { return uvw_; }
  void set_uvw(double3 uvw) { uvw_ = uvw; }

  const std::vector<float>& weight() const { return weight_; }
  void set_weight(std::vector<float> w) { weight_ = std::move(w); }

  const std::vector<float>& sigma() const { return sigma_; }
  void set_sigma(std::vector<float> s) { sigma_ = std::move(s); }

  const std::vector<VisSample>& visibilities() const { return vis_; }
  std::vector<VisSample>& visibilities() { return vis_; }
  void add_visibility(int chan, int pol, cufftComplex Vo, bool flagged = false) {
    vis_.push_back({chan, pol, Vo, {0.f, 0.f}, {0.f, 0.f}, flagged, 1.f, 1.f});
  }
  void add_visibility(int chan, int pol, cufftComplex Vo, cufftComplex Vm,
                     cufftComplex Vr, bool flagged = false) {
    vis_.push_back({chan, pol, Vo, Vm, Vr, flagged, 1.f, 1.f});
  }
  /** Add visibility with weight from MS; imaging_weight set equal to weight. */
  void add_visibility(int chan, int pol, cufftComplex Vo, cufftComplex Vm,
                     cufftComplex Vr, float weight, bool flagged = false) {
    vis_.push_back({chan, pol, Vo, Vm, Vr, flagged, weight, weight});
  }
  /** Set imaging weights for all visibilities (e.g. after weighting scheme). */
  void set_imaging_weights(const float* imaging_weights, size_t n) {
    for (size_t i = 0; i < vis_.size() && i < n; i++) vis_[i].imaging_weight = imaging_weights[i];
  }
  /** Reset imaging_weight = weight for all visibilities (e.g. restoreWeights). */
  void restore_imaging_weights() {
    for (auto& v : vis_) v.imaging_weight = v.weight;
  }

  void set_data_desc_id(int id) { data_desc_id_ = id; }
  void set_time(double t) { time_ = t; }

 private:
  int data_desc_id_{0};
  double time_{0.0};
  double3 uvw_{0.0, 0.0, 0.0};
  std::vector<float> weight_;
  std::vector<float> sigma_;
  std::vector<VisSample> vis_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_TIME_SAMPLE_H
