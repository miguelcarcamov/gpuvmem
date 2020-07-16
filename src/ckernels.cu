#include "ckernels.cuh"

template <class T>
__host__ __device__ T EllipticalGaussian2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T angle)
{
        T x_i = x-x0;
        T y_i = y-y0;
        T cos_angle, sin_angle;
        sincos(angle, &sin_angle, &cos_angle);
        T sin_angle_2 = sin(2.0*angle);
        T a = (cos_angle*cos_angle)/(2.0*sigma_x*sigma_x) + (sin_angle*sin_angle)/(2.0*sigma_y*sigma_y);
        T b = sin_angle_2/(2.0*sigma_x*sigma_x) - sin_angle_2/(2.0*sigma_y*sigma_y);
        T c = (sin_angle*sin_angle)/(2.0*sigma_x*sigma_x) + (cos_angle*cos_angle)/(2.0*sigma_y*sigma_y);
        T G = amp*exp(-a*x_i*x_i - b*x_i*y_i - c*y_i*y_i);

        return G;
}

template <class T>
__host__ __device__ T Gaussian2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T w, T alpha)
{
        T x_i = x-x0;
        T y_i = y-y0;

        T num_x = pow(x_i, alpha);
        T num_y = pow(y_i, alpha);

        T den_x = 2.0*pow(w*sigma_x,alpha);
        T den_y = 2.0*pow(w*sigma_y,alpha);

        T val_x = num_x/den_x;
        T val_y = num_y/den_y;
        T G = amp*exp(-val_x-val_y);

        return G;
}

template <class T>
__host__ __device__ T Gaussian1D(T amp, T x, T x0, T sigma, T w, T alpha)
{
        T x_i = x-x0;
        T val = abs(x_i)/(w*sigma);
        T val_alpha = pow(val, alpha);
        T G = amp*exp(-val_alpha);

        return G;
}

template <class T>
__host__ __device__ T Sinc1D(T amp, T x, T x0, T sigma, T w)
{
        T s = amp*sinc((x-x0)/(w*sigma));
        return s;
}

template <class T>
__host__ __device__ T GaussianSinc1D(T amp, T x, T x0, T sigma, T w1, T w2, T alpha)
{
        return amp*Gaussian1D(1.0, x, x0, sigma, w1, alpha)*Sinc1D(1.0, x, x0, sigma, w2);
}


template <class T>
__host__ __device__ T Sinc2D(T amp, T x, T x0, T y, T y0, T sigma_x, T sigma_y, T w)
{
        T s_x = Sinc1D(1.0, x, x0, sigma_x, w);
        T s_y = Sinc1D(1.0, y, y0, sigma_y, w);
        return amp*s_x*s_y;
}

template <class T>
__host__ __device__ T GaussianSinc2D(T amp, T x, T y, T x0, T y0, T sigma_x, T sigma_y, T w1, T w2, T alpha)
{
        T G = Gaussian2D(1.0, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        T S = Sinc2D(1.0, x, x0, y, y0, sigma_x, sigma_y, w2);
        return amp*G*S;
}

template <class Tdx, class T>
CKernel<Tdx, T>::CKernel()
{
        this->M = 6;
        this->N = 6;
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2;

};

template <class Tdx, class T>
CKernel<Tdx, T>::CKernel(Tdx dx, Tdx dy, int M, int N)
{
        this->M = M;
        this->N = N;
        this->dx = dx;
        this->dy = dy;
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2;
        this->kernel.reserve(M*N);
};

template <class Tdx, class T>
CKernel<Tdx, T>::CKernel(Tdx dx, Tdx dy, T w1, T w2, T alpha, int M, int N)
{
        this->M = M;
        this->N = N;
        this->dx = dx;
        this->dy = dy;
        this->w1 = w1;
        this->w2 = w2;
        this->alpha = alpha;
        this->kernel.reserve(M*N);
};

template <class Tdx, class T>
T CKernel<Tdx, T>::getdx()
{
        return this->dx;
};

template <class Tdx, class T>
T CKernel<Tdx, T>::getdy()
{
        return this->dy;
};

template <class Tdx, class T>
int2 CKernel<Tdx, T>::getMN()
{
        int2 val;
        val.x = this->M;
        val.y = this->N;
        return val;
};

template <class Tdx, class T>
T CKernel<Tdx, T>::getW1()
{
        return this->w1;
};

template <class Tdx, class T>
T CKernel<Tdx, T>::getW2()
{
        return this->w2;
}

template <class Tdx, class T>
T CKernel<Tdx, T>::getAlpha()
{
        return this->alpha;
};

template <class Tdx, class T>
std::vector<T>  CKernel<Tdx, T>::getCPUKernel()
{
        return this->kernel;
};

template <class Tdx, class T>
T* CKernel<Tdx, T>::getGPUKernel()
{
        T *gpu_kernel;
        gpuErrchk(cudaMemcpy(gpu_kernel, this->kernel.data(), sizeof(T) * this->kernel.size(), cudaMemcpyHostToDevice));
        return gpu_kernel;
};

template <class Tdx, class T>
void CKernel<Tdx, T>::setdxdy(Tdx dx, Tdx dy)
{
        this->dx = dx;
        this->dy = dy;
};

template <class Tdx, class T>
void CKernel<Tdx, T>::setMN(int M, int N)
{
        this->M = M;
        this->N = N;
};

template <class Tdx, class T>
void CKernel<Tdx, T>::setW1(T w1)
{
        this->w1 = w1;
};

template <class Tdx, class T>
void CKernel<Tdx, T>::setW2(T w2)
{
        this->w2 = w2;
};

template <class Tdx, class T>
void CKernel<Tdx, T>::setAlpha(T alpha)
{
        this->alpha=alpha;
};
