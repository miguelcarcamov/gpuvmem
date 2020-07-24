#include "ckernels.cuh"

__host__ __device__ float EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle)
{
        float x_i = x-x0;
        float y_i = y-y0;
        float cos_angle, sin_angle;
        sincos(angle, &sin_angle, &cos_angle);
        float sin_angle_2 = sin(2.0*angle);
        float a = (cos_angle*cos_angle)/(2.0*sigma_x*sigma_x) + (sin_angle*sin_angle)/(2.0*sigma_y*sigma_y);
        float b = sin_angle_2/(2.0*sigma_x*sigma_x) - sin_angle_2/(2.0*sigma_y*sigma_y);
        float c = (sin_angle*sin_angle)/(2.0*sigma_x*sigma_x) + (cos_angle*cos_angle)/(2.0*sigma_y*sigma_y);
        float G = amp*exp(-a*x_i*x_i - b*x_i*y_i - c*y_i*y_i);

        return G;
}


__host__ __device__ float Gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float x_i = x-x0;
        float y_i = y-y0;

        float num_x = pow(x_i, alpha);
        float num_y = pow(y_i, alpha);

        float den_x = 2.0*pow(w*sigma_x,alpha);
        float den_y = 2.0*pow(w*sigma_y,alpha);

        float val_x = num_x/den_x;
        float val_y = num_y/den_y;
        float G = amp*exp(-val_x-val_y);

        return G;
}

__host__ __device__ float Gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha)
{
        float x_i = x-x0;
        float val = abs(x_i)/(w*sigma);
        float val_alpha = pow(val, alpha);
        float G = amp*exp(-val_alpha);

        return G;
}


__host__ __device__ float Sinc1D(float amp, float x, float x0, float sigma, float w)
{
        float s = 1.0f/*amp*sinc((x-x0)/(w*sigma))*/;
        return s;
}

__host__ __device__ float GaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha)
{
        return amp*Gaussian1D(1.0, x, x0, sigma, w1, alpha)*Sinc1D(1.0, x, x0, sigma, w2);
}


__host__ __device__ float Sinc2D(float amp, float x, float x0, float y, float y0, float sigma_x, float sigma_y, float w)
{
        float s_x = Sinc1D(1.0, x, x0, sigma_x, w);
        float s_y = Sinc1D(1.0, y, y0, sigma_y, w);
        return amp*s_x*s_y;
}

__host__ __device__ float GaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha)
{
        float G = Gaussian2D(1.0, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        float S = Sinc2D(1.0, x, x0, y, y0, sigma_x, sigma_y, w2);
        return amp*G*S;
}


CKernel::CKernel()
{
        this->M = 6;
        this->N = 6;
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2;

};


CKernel::CKernel(float dx, float dy, int M, int N)
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


CKernel::CKernel(float dx, float dy, float w1, float w2, float alpha, int M, int N)
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

float CKernel::getdx()
{
        return this->dx;
};

float CKernel::getdy()
{
        return this->dy;
};

int2 CKernel::getMN()
{
        int2 val;
        val.x = this->M;
        val.y = this->N;
        return val;
};

float CKernel::getW1()
{
        return this->w1;
};

float CKernel::getW2()
{
        return this->w2;
}

float CKernel::getAlpha()
{
        return this->alpha;
};

std::vector<float>  CKernel::getCPUKernel()
{
        return this->kernel;
};

float* CKernel::getGPUKernel()
{
        float *gpu_kernel;
        checkCudaErrors(cudaMemcpy(gpu_kernel, this->kernel.data(), sizeof(float) * this->kernel.size(), cudaMemcpyHostToDevice));
        return gpu_kernel;
};

void CKernel::setdxdy(float dx, float dy)
{
        this->dx = dx;
        this->dy = dy;
};

void CKernel::setMN(int M, int N)
{
        this->M = M;
        this->N = N;
};

void CKernel::setW1(float w1)
{
        this->w1 = w1;
};

void CKernel::setW2(float w2)
{
        this->w2 = w2;
};

void CKernel::setAlpha(float alpha)
{
        this->alpha=alpha;
};

/*
float CKernel::run(float deltau, float deltav){
        switch (mode) {
                case 0:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 1:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 2:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 3:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 4:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 5:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                case 6:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
                default:
                        return this->EllipticalGaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float angle);
                        break;
        };
}*/

void CKernel(int mode){
        
}