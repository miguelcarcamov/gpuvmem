#include "gridding.cuh"

extern double deltau, deltav;
extern long M, N;
extern int num_gpus;

Gridding::Gridding() { this->threads = 1; };

void Gridding::applyCriteria(Visibilities *v) {
  double3 valzero;
  cufftComplex complexValZero;
  valzero.x = 0.0;
  valzero.y = 0.0;
  valzero.z = 0.0;

  complexValZero.x = 0.0f;
  complexValZero.y = 0.0f;
  for (int d = 0; d < v->getNDatasets(); d++) {
    do_gridding(v->getMSDataset()[d].fields, &v->getMSDataset()[d].data, deltau,
                deltav, M, N, NULL, this->threads);
  }
};

Gridding::Gridding(int threads) {
  if (threads != 1 && threads >= 1)
    this->threads = threads;
  else if (threads != 1)
    printf("Number of threads set to 1\n");
};

void Gridding::configure(void *params) {
  int *threads = (int *)params;
  printf("Number of threads = %d\n", *threads);
  if (*threads != 1 && *threads >= 1)
    this->threads = *threads;
  else if (*threads != 1)
    printf("Number of threads set to 1\n");
};

namespace {
Filter *CreateGridding() { return new Gridding; }
const std::string GriddingId = "Gridding";
const bool RegisteredGridding =
    registerCreationFunction<Filter, std::string>(GriddingId, CreateGridding);
};  // namespace
