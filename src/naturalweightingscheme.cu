#include "naturalweightingscheme.cuh"

NaturalWeightingScheme::NaturalWeightingScheme(){};

void NaturalWeightingScheme::apply(std::vector<MSDataset> d)
{
        return;
};

namespace {
WeightingScheme* CreateWeightingScheme()
{
        return new NaturalWeightingScheme;
}
const int NaturalWeightingSchemeId = 0;
const bool RegisteredWeightingScheme = Singleton<WeightingSchemeFactory>::Instance().RegisterWeightingScheme(NaturalWeightingSchemeId, CreateWeightingScheme);
};
