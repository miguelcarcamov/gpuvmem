#ifndef COMPLEXOPS_CUH
#define COMPLEXOPS_CUH
#include <cufft.h>

template <class T>
__host__ __device__ T complexZero()
{
        T zero = make_cuComplex(0.0, 0.0);
        return zero;
}

template <class T, class R>
__host__ __device__ T multComplexReal(T c1, R c2)
{

        T result;

        result.x = c1.x * c2;
        result.y = c1.y * c2;

        return result;

}

template <class T>
__host__ __device__ T multComplexComplex(T c1, T c2)
{

        T result;

        result.x = (c1.x * c2.x) - (c1.y * c2.y);
        result.y = (c1.x * c2.y) + (c1.y * c2.x);
        return result;

}

template <class T, class R>
__host__ __device__ T divComplexReal(T c1, R c2)
{

        T result;

        result.x = c1.x / c2;
        result.y = c1.y / c2;

        return result;

}


template <class T, class R>
__host__ __device__ T divComplexComplex(T c1, T c2)
{

        T result;
        R r, den;

        if(fabsf(c2.x) >= fabsf(c2.y))
        {
                r = c2.y/c2.x;
                den = c2.x+r*c2.y;
                result.x = (c1.x+r*c1.y)/den;
                result.y = (c1.y-r*c1.x)/den;

        }else{
                r = c2.x/c2.y;
                den = c2.y+r*c2.x;
                result.x = (c1.x*r+c1.y)/den;
                result.y = (c1.y*r-c1.x)/den;

        }

        return result;

}

template <class T>
__host__ __device__ T addComplexComplex(T c1, T c2)
{

        T result;

        result.x = c1.x + c2.x;
        result.y = c1.y + c2.y;
        return result;

}

template <class T>
__host__ __device__ T subComplexComplex(T c1, T c2)
{

        T result;

        result.x = c1.x - c2.x;
        result.y = c1.y - c2.y;
        return result;

}

template <class T>
__host__ __device__ T ConjComplex(T c1)
{

        T result;

        result.x = c1.x;
        result.y = -c1.y;
        return result;

}

template <class T>
__global__ void multArrayComplexComplex(T *c1, T *c2, int M, int N)
{
        const int i = threadIdx.y + blockDim.y * blockIdx.y;
        const int j = threadIdx.x + blockDim.x * blockIdx.x;

        if(i<M && j<N) {
                c1[N*i+j] = multComplexComplex<T>(c1[N*i+j], c2[N*i+j]);
        }
}

#endif
