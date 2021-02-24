#include "naturalweightingscheme.cuh"

NaturalWeightingScheme::NaturalWeightingScheme() : WeightingScheme(){};
NaturalWeightingScheme::NaturalWeightingScheme(int threads) : WeightingScheme(threads){};

void NaturalWeightingScheme::apply(std::vector<MSDataset>& d){};

namespace {
WeightingScheme* CreateWeightingScheme()
{
        return new NaturalWeightingScheme;
}

const std::string name = "Natural";
const bool RegisteredNaturalWeighting = registerCreationFunction<WeightingScheme, std::string>(name, CreateWeightingScheme);
};
