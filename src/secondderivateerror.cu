#include "classes/secondderivateerror.cuh"
#include "classes/optimizer.cuh"
#include "factory.cuh"
#include "functions.cuh"

void SecondDerivateError::calculateErrorImage(Image* I, Visibilities* v) {
  calculateErrors(I, fg_scale);
}

static Error* CreateSecondDerivateError() {
  return new SecondDerivateError();
}

static const bool SecondDerivateErrorRegistered =
    registerCreationFunction<Error, std::string>("SecondDerivateError",
                                                 CreateSecondDerivateError);
