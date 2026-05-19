#include "classes/objectivefunction.cuh"

void Fi::attachToObjectiveFunction(ObjectiveFunction* o) {
  if (o == nullptr) {
    grid_M_ = 0;
    grid_N_ = 0;
    grid_image_count_ = 0;
    z_weights_ptr_ = nullptr;
    n_z_weights_ = 0;
    return;
  }
  grid_M_ = o->getM();
  grid_N_ = o->getN();
  grid_image_count_ = o->getImageCount();
  z_weights_ptr_ = o->getRegularizationWeights();
  n_z_weights_ = o->getRegularizationWeightCount();
}
