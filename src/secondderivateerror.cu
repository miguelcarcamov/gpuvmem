#include <secondderivateerror.cuh>
#include <functions.cuh>

extern long N, M;

void SecondDerivateError::calculateErrorImage(Image *I, Visibilities *v)
{
        if(I->getImageCount() > 1)
                calculateErrors(I);
};

namespace {
Error* CreateSecondDerivateError()
{
        return new SecondDerivateError;
}
const std::string SecondDerivateErrorID = "SecondDerivateError";
const bool RegisteredSecondDerivateError = registerCreationFunction<Error, std::string>(SecondDerivateErrorID, CreateSecondDerivateError);
};
