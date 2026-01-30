#include "functions.cuh"
#include "objective_function/terms/regularizers/secondderivateerror.cuh"

extern long N, M;

void SecondDerivateError::calculateErrorImage(
    Image* I, std::vector<gpuvmem::ms::MSWithGPU>& datasets) {
  (void)datasets;
  if (I->getImageCount() > 1) {
    // Get fg_scale from Chi2 if available, otherwise use stored value
    float fg_scale_val = this->fg_scale;
    if (this->optimizer != NULL) {
      ObjectiveFunction* of = this->optimizer->getObjectiveFunction();
      if (of != NULL) {
        Fi* chi2 = of->getFiByName("Chi2");
        if (chi2 != NULL) {
          float chi2_fg_scale = chi2->getFgScale();
          // If Chi2 returns a non-zero value, use it (base class returns 0.0)
          if (chi2_fg_scale != 0.0f) {
            fg_scale_val = chi2_fg_scale;
          }
        }
      }
    }
    calculateErrors(I, fg_scale_val);
  }
};

namespace {
Error* CreateSecondDerivateError() {
  return new SecondDerivateError;
}
const std::string SecondDerivateErrorID = "SecondDerivateError";
const bool RegisteredSecondDerivateError =
    registerCreationFunction<Error, std::string>(SecondDerivateErrorID,
                                                 CreateSecondDerivateError);
};  // namespace
