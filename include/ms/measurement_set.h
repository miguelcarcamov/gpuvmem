#ifndef GPUVMEM_MS_MEASUREMENT_SET_H
#define GPUVMEM_MS_MEASUREMENT_SET_H

#include "ms/field.h"
#include "ms/measurement_set_metadata.h"

#include <memory>
#include <string>
#include <vector>

namespace gpuvmem {
namespace ms {

/** In-memory polarization storage: correlations (MS native) or Stokes. */
enum class StorageMode { Correlations, Stokes };

/**
 * Top-level container: shared MS metadata + list of fields.
 * Aligned with Measurement Set structure; decoupled from raw table layout.
 * When storage_mode() == Stokes, pol index = Stokes (I=0, Q=1, U=2, V=3);
 * writer requires Correlations to write to disk.
 */
class MeasurementSet {
 public:
  MeasurementSet() = default;
  explicit MeasurementSet(const std::string& name) : name_(name) {}

  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }

  MeasurementSetMetadata& metadata() { return metadata_; }
  const MeasurementSetMetadata& metadata() const { return metadata_; }

  Field& add_field(const FieldMetadata& meta) {
    fields_.emplace_back(meta);
    return fields_.back();
  }
  Field& field(size_t i) { return fields_.at(i); }
  const Field& field(size_t i) const { return fields_.at(i); }
  std::vector<Field>& fields() { return fields_; }
  const std::vector<Field>& fields() const { return fields_; }
  size_t num_fields() const { return fields_.size(); }

  StorageMode storage_mode() const { return storage_mode_; }
  void set_storage_mode(StorageMode m) { storage_mode_ = m; }

  /** When storage_mode() == Stokes, which Stokes (e.g. ["I","Q","U","V"]). */
  const std::vector<std::string>& stored_stokes_list() const {
    return stored_stokes_list_;
  }
  void set_stored_stokes_list(std::vector<std::string> s) {
    stored_stokes_list_ = std::move(s);
  }

 private:
  std::string name_;
  MeasurementSetMetadata metadata_;
  std::vector<Field> fields_;
  StorageMode storage_mode_{StorageMode::Correlations};
  std::vector<std::string> stored_stokes_list_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_MEASUREMENT_SET_H
