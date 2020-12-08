#include "naturalweightingscheme.cuh"

NaturalWeightingScheme::NaturalWeightingScheme(){};

void NaturalWeightingScheme::apply(std::vector<MSDataset>& d){};

namespace {
WeightingScheme* CreateWeightingScheme()
{
        return new NaturalWeightingScheme;
}

const std::string name = "Natural";
const bool RegisteredNaturalWeighting = registerCreationFunction<WeightingScheme, std::string>(name, CreateWeightingScheme);
};
