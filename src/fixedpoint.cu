#include "fixedpoint.cuh"


__host__ float tolFunction(std::vector<float> x1, std::vector<float> x0)
{
        std::vector<float> tolVector;
        for(int i=0; i<x1.size(); i++) {
                tolVector.push_back(abs(x1[i]-x0[i])/abs(x0[i]));
        }

        float max = *max_element(tolVector.begin(), tolVector.end());

        return max;
};

// The algorithm will return a vector with the solutions
__host__ std::vector<float> fixedPointOpt(std::vector<float> guess, std::vector<float> (*optf)(std::vector<float>, Synthesizer*), float tol, int iterations, Synthesizer *synthesizer)
{
        std::vector<float> x0 = guess;
        std::vector<float> x1(x0.size(), 0.0);
        int it = 0;

        while(it < iterations && tolFunction(x1, x0) < tol) {
                x1 = optf(x0, synthesizer);
                it++;
        }

        return x1;

};
