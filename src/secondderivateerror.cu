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
const int SecondDerivateErrorID = 0;
const bool RegisteredSecondDerivateError = Singleton<ErrorFactory>::Instance().RegisterError(SecondDerivateErrorID, CreateSecondDerivateError);
};
