#ifndef OPTIMIZATION_PROJECTION_HH
#define OPTIMIZATION_PROJECTION_HH

#include <memory>
#include <vector>

/**
 * Optimization projection (mirrors Pyralysis `optimization.projection.Projection` + `f1dim`).
 *
 * Implementations live in `src/optimization/projection.cu`.
 */
class Projection {
 public:
  virtual ~Projection();

  /** MEM positivity kernel `eta` when using fused `newP` / `evaluateXt` (< 0 typical); default unused. */
  virtual float positivityEta() const;

  /** Per-image reference level for `evaluateXt` (legacy `initial_values[image]`). */
  virtual float referenceValue(int image_index) const;

  /**
   * Optional device pass on one image plane after a trial update (`p` or `xt`),
   * analogous to Pyralysis `projection(parameter)`.
   */
  virtual void applyToImagePlane(float* buffer, long N, long M, int image, unsigned blocks_x,
                                 unsigned blocks_y, unsigned threads_x, unsigned threads_y) const;
};

class NoProjection : public Projection {};

class PositivityProjection : public Projection {
 public:
  explicit PositivityProjection(float eta, std::vector<float> xt_reference_per_image);

  float positivityEta() const override;
  float referenceValue(int image_index) const override;

 private:
  float eta_;
  std::vector<float> xt_reference_;
};

enum class ScalarProjectionOp : int {
  EqualTo = 0,
  NotEqualTo = 1,
  GreaterThan = 2,
  GreaterThanEqualTo = 3,
  LessThan = 4,
  LessThanEqualTo = 5,
};

class ScalarReplaceProjection : public Projection {
 protected:
  ScalarReplaceProjection(float compared_value, float replacement_value, ScalarProjectionOp op);

 public:
  void applyToImagePlane(float* buffer, long N, long M, int image, unsigned blocks_x,
                         unsigned blocks_y, unsigned threads_x, unsigned threads_y) const override;

  float comparedValue() const;
  float replacementValue() const;
  ScalarProjectionOp op() const;

 private:
  float compared_value_;
  float replacement_value_;
  ScalarProjectionOp op_;
};

class EqualTo : public ScalarReplaceProjection {
 public:
  EqualTo(float compared_value, float replacement_value);
};

class NotEqualTo : public ScalarReplaceProjection {
 public:
  NotEqualTo(float compared_value, float replacement_value);
};

class GreaterThan : public ScalarReplaceProjection {
 public:
  GreaterThan(float compared_value, float replacement_value);
};

class GreaterThanEqualTo : public ScalarReplaceProjection {
 public:
  GreaterThanEqualTo(float compared_value, float replacement_value);
};

class LessThan : public ScalarReplaceProjection {
 public:
  LessThan(float compared_value, float replacement_value);
};

class LessThanEqualTo : public ScalarReplaceProjection {
 public:
  LessThanEqualTo(float compared_value, float replacement_value);
};

/** Chain projections in order (Pyralysis `CompositeProjection`). */
class CompositeProjection : public Projection {
 public:
  CompositeProjection();
  explicit CompositeProjection(std::vector<std::unique_ptr<Projection>> parts);

  void append(std::unique_ptr<Projection> p);

  void applyToImagePlane(float* buffer, long N, long M, int image, unsigned blocks_x,
                         unsigned blocks_y, unsigned threads_x, unsigned threads_y) const override;

  float positivityEta() const override;
  float referenceValue(int image_index) const override;

 private:
  std::vector<std::unique_ptr<Projection>> parts_;
};

#endif
