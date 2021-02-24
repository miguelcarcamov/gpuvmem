#ifndef BRIGGSWEIGHTINGSCHEME_CUH
#define BRIGGSWEIGHTINGSCHEME_CUH

#include "framework.cuh"
#include "functions.cuh"

extern int gridding;
extern long M;
extern long N;
extern double deltau;
extern double deltav;

class BriggsWeightingScheme : public WeightingScheme
{
public:
BriggsWeightingScheme();
BriggsWeightingScheme(int threads);
void apply(std::vector<MSDataset>& d);
void configure(void *params);
float getRobustParam();
void setRobustParam(float robust_param);

private:
float robust_param;
};

#endif
