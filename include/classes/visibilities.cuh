#ifndef VISIBILITIES_CUH
#define VISIBILITIES_CUH

#include "MSFITSIO.cuh"

class Visibilities {
 public:
  void setMSDataset(std::vector<MSDataset>& d) { this->datasets = d; };
  void setTotalVisibilities(int t) { this->total_visibilities = t; };

  void setNDatasets(int t) { this->ndatasets = t; };

  void setMaxNumberVis(int t) { this->max_number_vis = t; };

  std::vector<MSDataset> getMSDataset() { return this->datasets; };
  int getTotalVisibilities() { return this->total_visibilities; };

  int getMaxNumberVis() { return this->max_number_vis; };

  int getNDatasets() { return this->ndatasets; };

  void applyWeightingScheme(WeightingScheme* scheme) {
    scheme->apply(this->datasets);
  }

 private:
  std::vector<MSDataset> datasets;
  int ndatasets;
  int total_visibilities;
  int max_number_vis;
};

#endif
