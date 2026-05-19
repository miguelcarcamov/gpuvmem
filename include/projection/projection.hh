#ifndef GPUVMEM_PROJECTION_HH
#define GPUVMEM_PROJECTION_HH

#include <memory>
#include <vector>

/** Where `applyToImagePlane` runs (Pyralysis: linear step then `projection(parameter)`). */
enum class ProjectionApplyContext {
  kParameterStep,      /**< After `p += α d` along the search line (iterate update). */
  kLineSearch1dSample  /**< After `xt = pcom + x·xicom` for 1D line objective `f(α)`. */
};

/**
 * Line-search projection (mirrors Pyralysis `optimization.projection.Projection` + `f1dim`).
 *
 * Implementations live in `src/projection/projection.cu`.
 */
class Projection {
 public:
  virtual ~Projection();

  /** Legacy MEM `η` (reserved; fused kernels removed). */
  virtual float positivityEta() const;

  /** Per-image reference level along the line (legacy `initial_values[image]`).
   *  Falls back to `minimalValue` when no reference vector entry exists. */
  virtual float referenceValue(int image_index) const;

  /** Per-image minimum pixel floor (legacy `Image::minimal_pixel_values`). */
  virtual float minimalValue(int image_index) const;

  /**
   * Device pass on one image plane after the linear trial (Pyralysis `projection(parameter)`).
   */
  virtual void applyToImagePlane(float* buffer, long N, long M, int image, unsigned blocks_x,
                                 unsigned blocks_y, unsigned threads_x, unsigned threads_y,
                                 ProjectionApplyContext ctx) const;
};

class NoProjection : public Projection {};

/**
 * MEM positivity: linear trial uses `newPNoPositivity` / `evaluateXtNoPositivity`;
 * this projection clamps plane 0 to `minimalValue` (parameter step) or `referenceValue`
 * (1D line sample), matching legacy MEM behavior for the primary plane.
 */
class PositivityProjection : public Projection {
 public:
  PositivityProjection(float eta, std::vector<float> xt_reference_per_image,
                       std::vector<float> minimal_pixel_values_per_image);

  float positivityEta() const override;
  float referenceValue(int image_index) const override;
  float minimalValue(int image_index) const override;

  void applyToImagePlane(float* buffer, long N, long M, int image, unsigned blocks_x,
                         unsigned blocks_y, unsigned threads_x, unsigned threads_y,
                         ProjectionApplyContext ctx) const override;

 private:
  float eta_;
  std::vector<float> xt_reference_;
  std::vector<float> minimal_pixel_;
};

#endif
