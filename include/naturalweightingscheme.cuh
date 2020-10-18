#ifndef NATURALWEIGHTINGSCHEME_CUH
#define NATURALWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

class NaturalWeightingScheme : public WeightingScheme
{
public:
NaturalWeightingScheme();
void apply(std::vector<MSDataset> d);
};

#endif
