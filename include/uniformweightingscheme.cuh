#ifndef UNIFORMWEIGHTINGSCHEME_CUH
#define UNIFORMWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

extern int gridding;
extern long M;
extern long N;
extern double deltau;
extern double deltav;

class UniformWeightingScheme : public WeightingScheme
{
public:
UniformWeightingScheme();
UniformWeightingScheme(int threads);
void apply(std::vector<MSDataset>& d);
void configure(void *params){};
};

#endif
