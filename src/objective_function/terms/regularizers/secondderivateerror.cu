#include "classes/objectivefunction.cuh"
#include "objective_function/terms/regularizers/secondderivateerror.cuh"
#include "functions.cuh"

void SecondDerivateError::calculateErrorImage(Image* I, Visibilities* v) {
  (void)v;
  float fg_scale_val = this->fg_scale;
  if (this->optimizer != nullptr) {
    ObjectiveFunction* of = this->optimizer->getObjectiveFunction();
    if (of != nullptr) {
      Fi* chi2 = of->getFiByName("Chi2");
      if (chi2 != nullptr) {
        float chi2_fg_scale = chi2->getFgScale();
        if (chi2_fg_scale != 0.0f)
          fg_scale_val = chi2_fg_scale;
      }
    }
  }
  calculateErrors(I, fg_scale_val);
}

namespace {
Error* CreateSecondDerivateError() {
  return new SecondDerivateError;
}
const std::string SecondDerivateErrorID = "SecondDerivateError";
const bool RegisteredSecondDerivateError =
    registerCreationFunction<Error, std::string>(SecondDerivateErrorID,
                                                 CreateSecondDerivateError);
};  // namespace
