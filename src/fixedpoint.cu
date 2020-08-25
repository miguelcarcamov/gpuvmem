#include "fixedpoint.cuh"
// This can be any function to calculate the error
__host__ float tolFunction(std::vector<float> x1, std::vector<float> x0)
{
        std::vector<float> tolVector(x1.size()-1, 0.0);
        float tol;
        for(int i=1; i<=tolVector.size(); i++) {
                if(fabsf(x0[i]) == 0.0f) {
                        tol = 0.0f;
                }else{
                        tol = fabsf(x1[i]-x0[i])/fabsf(x0[i]);
                }
                tolVector[i-1] = tol;
        }

        float max = *max_element(tolVector.begin(), tolVector.end());
        std::cout << "Tolerance: " << max << std::endl;
        return max;
};

__host__ bool areThereAnyGreaterThanOne(std::vector<float> x1)
{
        bool var = false;
        for(int i=1; i<x1.size(); i++) {
                if(x1[i]>=1.0f) {
                        var = true;
                        break;
                }
        }
        return var;
}

// The algorithm will return a vector with the solutions
__host__ std::vector<float> fixedPointOpt(std::vector<float> guess, std::vector<float> (*optf)(std::vector<float>, Synthesizer*), float tol, int iterations, Synthesizer *synthesizer)
{
        std::vector<float> x0 = guess;
        std::vector<float> x1(x0.size(), 0.0);
        int it = 0;
        float c_tol = tolFunction(x1, x0);
        while(it < iterations &&  c_tol > tol) {
                if(it > 1)
                        x0 = x1;
                x1 = optf(x0, synthesizer);
                for(int i = 0; i < x1.size(); i++)
                {
                        std::cout << x1[i] << std::endl;
                }
                c_tol = tolFunction(x1, x0);
                if(areThereAnyGreaterThanOne(x1)) {
                        break;
                }
                it++;
        }
        std::cout << "Fixed point iterations: " << it << std::endl;
        return x1;

}
