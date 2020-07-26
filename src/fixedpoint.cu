// This can be any function to calculate the error
template <class T>
__host__ T tolFunction(std::vector<T> x1, std::vector<T> x0)
{
        std::vector<T> tolVector;
        for(int i=0; i<x1.size(); i++) {
                tolVector.push_back(abs(x1[i]-x0[i])/abs(x0[i]));
        }

        T max = *max_element(tolVector.begin(), tolVectorv1.end());

        return max;
}

// The algorithm will return a vector with the solutions
template <class T>
__host__ std::vector<T> fixedPointOpt(std::vector<T> guess, void (*optf)(void *), T tol, int iterations)
{
        std::vector<T> x0 = guess;
        std::vector<T> x1(x0.size(), 0.0);
        int it = 0;

        while(it < iterations && tolFunction<T>(x1, x0) < tol) {
                x1 = optf(x0);
                it++;
        }

        return x1;

}
