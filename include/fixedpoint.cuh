#ifndef FIXEDPOINT_CUH
#define FIXEDPOINT_CUH

#include "framework.cuh"

__host__ float tolFunction(std::vector<float> x1, std::vector<float> x0);
__host__ std::vector<float> fixedPointOpt(std::vector<float> guess, std::vector<float> (*optf)(std::vector<float>, Synthesizer*), float tol, int iterations, Synthesizer *synthesizer);

#endif
