#ifndef GPUVMEM_MS_MEASUREMENT_SET_METADATA_H
#define GPUVMEM_MS_MEASUREMENT_SET_METADATA_H

#include "ms/metadata.h"

#include <map>
#include <stdexcept>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * MS-level shared metadata: Spectral Windows, Data Descriptions (and optionally
 * Polarization, Antenna). Stored once; no per-field duplication of frequencies.
 */
class MeasurementSetMetadata {
 public:
  void add_spectral_window(SpectralWindow sw) {
    int id = sw.spectral_window_id();
    spectral_windows_.push_back(std::move(sw));
    spw_id_to_index_[id] = spectral_windows_.size() - 1;
  }
  const SpectralWindow& spectral_window(int spw_id) const {
    auto it = spw_id_to_index_.find(spw_id);
    if (it == spw_id_to_index_.end())
      throw std::out_of_range("SpectralWindow id not found");
    return spectral_windows_.at(it->second);
  }
  SpectralWindow* find_spectral_window(int spw_id) {
    auto it = spw_id_to_index_.find(spw_id);
    if (it == spw_id_to_index_.end()) return nullptr;
    return &spectral_windows_.at(it->second);
  }
  const SpectralWindow* find_spectral_window(int spw_id) const {
    auto it = spw_id_to_index_.find(spw_id);
    if (it == spw_id_to_index_.end()) return nullptr;
    return &spectral_windows_.at(it->second);
  }
  const std::vector<SpectralWindow>& spectral_windows() const {
    return spectral_windows_;
  }

  void add_data_description(DataDescription dd) {
    int id = dd.data_desc_id();
    data_descriptions_.push_back(std::move(dd));
    data_desc_id_to_index_[id] = data_descriptions_.size() - 1;
  }
  const DataDescription& data_description(int data_desc_id) const {
    auto it = data_desc_id_to_index_.find(data_desc_id);
    if (it == data_desc_id_to_index_.end())
      throw std::out_of_range("DataDescription id not found");
    return data_descriptions_.at(it->second);
  }
  DataDescription* find_data_description(int data_desc_id) {
    auto it = data_desc_id_to_index_.find(data_desc_id);
    if (it == data_desc_id_to_index_.end()) return nullptr;
    return &data_descriptions_.at(it->second);
  }
  const DataDescription* find_data_description(int data_desc_id) const {
    auto it = data_desc_id_to_index_.find(data_desc_id);
    if (it == data_desc_id_to_index_.end()) return nullptr;
    return &data_descriptions_.at(it->second);
  }
  const std::vector<DataDescription>& data_descriptions() const {
    return data_descriptions_;
  }

  void add_polarization(Polarization pol) {
    int id = pol.polarization_id();
    polarizations_.push_back(std::move(pol));
    pol_id_to_index_[id] = polarizations_.size() - 1;
  }
  const Polarization& polarization(int pol_id) const {
    auto it = pol_id_to_index_.find(pol_id);
    if (it == pol_id_to_index_.end())
      throw std::out_of_range("Polarization id not found");
    return polarizations_.at(it->second);
  }
  Polarization* find_polarization(int pol_id) {
    auto it = pol_id_to_index_.find(pol_id);
    if (it == pol_id_to_index_.end()) return nullptr;
    return &polarizations_.at(it->second);
  }
  const Polarization* find_polarization(int pol_id) const {
    auto it = pol_id_to_index_.find(pol_id);
    if (it == pol_id_to_index_.end()) return nullptr;
    return &polarizations_.at(it->second);
  }
  const std::vector<Polarization>& polarizations() const {
    return polarizations_;
  }

  void add_antenna(Antenna ant) { antennas_.push_back(std::move(ant)); }
  const Antenna& antenna(size_t i) const { return antennas_.at(i); }
  Antenna* antenna_ptr(size_t i) {
    return i < antennas_.size() ? &antennas_[i] : nullptr;
  }
  const std::vector<Antenna>& antennas() const { return antennas_; }
  std::vector<Antenna>& antennas() { return antennas_; }
  size_t num_antennas() const { return antennas_.size(); }

  void build_index() {
    spw_id_to_index_.clear();
    data_desc_id_to_index_.clear();
    pol_id_to_index_.clear();
    for (size_t i = 0; i < spectral_windows_.size(); ++i)
      spw_id_to_index_[spectral_windows_[i].spectral_window_id()] = i;
    for (size_t i = 0; i < data_descriptions_.size(); ++i)
      data_desc_id_to_index_[data_descriptions_[i].data_desc_id()] = i;
    for (size_t i = 0; i < polarizations_.size(); ++i)
      pol_id_to_index_[polarizations_[i].polarization_id()] = i;
  }

 private:
  std::vector<SpectralWindow> spectral_windows_;
  std::vector<DataDescription> data_descriptions_;
  std::vector<Polarization> polarizations_;
  std::vector<Antenna> antennas_;
  std::map<int, size_t> spw_id_to_index_;
  std::map<int, size_t> data_desc_id_to_index_;
  std::map<int, size_t> pol_id_to_index_;
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_MEASUREMENT_SET_METADATA_H
