#ifndef RADIALWEIGHTINGSCHEME_CUH
#define RADIALWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

extern int gridding;
extern long M;
extern long N;
extern double deltau;
extern double deltav;

class RadialWeightingScheme : public WeightingScheme
{
public:
RadialWeightingScheme();
void apply(std::vector<MSDataset>& d);
void configure(void *params){};
};

#endif
