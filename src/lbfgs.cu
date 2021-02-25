/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */

#include "lbfgs.cuh"

extern long M;
extern long N;

extern ObjectiveFunction *testof;
extern Image *I;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int verbose_flag;
extern int flag_opt;

#define EPS 1.0e-10

#define FREEALL cudaFree(d_y); cudaFree(d_s); cudaFree(xi); cudaFree(xi_old); cudaFree(p_old); cudaFree(norm_vector);

__host__ int LBFGS::getK(){
        return this->K;
};

__host__ void LBFGS::setK(int K){
        this->K = K;
};

__host__ void LBFGS::allocateMemoryGpu()
{
        checkCudaErrors(cudaMalloc((void**)&d_y, sizeof(float)*M*N*K*image->getImageCount()));
        checkCudaErrors(cudaMemset(d_y, 0, sizeof(float)*M*N*K*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&d_s, sizeof(float)*M*N*K*image->getImageCount()));
        checkCudaErrors(cudaMemset(d_s, 0, sizeof(float)*M*N*K*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&p_old, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(p_old, 0, sizeof(float)*M*N*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&xi, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(xi, 0, sizeof(float)*M*N*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&xi_old, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(xi_old, 0, sizeof(float)*M*N*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&norm_vector, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(norm_vector, 0, sizeof(float)*M*N*image->getImageCount()));
};
__host__ void LBFGS::deallocateMemoryGpu()
{
        FREEALL
};

__host__ void LBFGS::optimize()
{
        std::cout << "\n\nStarting Lbfgs\n" << std::endl;
        double start, end;
        I = image;
        flag_opt = this->flag;
        allocateMemoryGpu();
        testof = of;
        if(configured) {
                of->configure(N, M, image->getImageCount());
                configured = 0;
        }

        fp = of->calcFunction(image->getImage());
        if(verbose_flag) {
                std::cout << "Starting function value = " << std::setprecision(4) << std::fixed << fp << std::endl;
        }
        of->calcGradient(image->getImage(),xi, 0);


        //checkCudaErrors(cudaMemcpy(p_old, image->getImage(), sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
        //checkCudaErrors(cudaMemcpy(xi_old, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));

        for(int i=0; i < image->getImageCount(); i++)
        {
                searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(xi, M, N, i); //Search direction
                checkCudaErrors(cudaDeviceSynchronize());
        }

        ////////////////////////////////////////////////////////////////
        for(int i=1; i <= this->total_iterations; i++) {
                start = omp_get_wtime();
                this->current_iteration = i;
                this->max_per_it = 0.0f;
                if(verbose_flag) {
                        std::cout << "\n\n********** Iteration "<< i <<" **********\n" << std::endl;
                }

                checkCudaErrors(cudaMemcpy(p_old, image->getImage(), sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaMemcpy(xi_old, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));

                linmin(image->getImage(), xi, &fret, NULL);

                if((fp - fret)/std::max({fabsf(fret), fabsf(fp), 1.0f}) <= this->ftol){
                        std::cout << "Exit due to tolerance" << std::endl;
                        of->calcFunction(I->getImage());
                        deallocateMemoryGpu();
                        return;
                }

                for(int i=0; i < image->getImageCount(); i++)
                {
                        normArray<<<numBlocksNN, threadsPerBlockNN>>>(norm_vector, xi, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }

                this->max_per_it = std::max(this->max_per_it, deviceMaxReduce(norm_vector, M*N*image->getImageCount(), threadsPerBlockNN.x * threadsPerBlockNN.y));

                if(this->max_per_it <= this->gtol) {
                        std::cout << "Exit due to gnorm ~ 0" << std::endl;
                        of->calcFunction(image->getImage());
                        deallocateMemoryGpu();
                        return;
                }


                fp= of->calcFunction(image->getImage());
                if(verbose_flag) {
                        std::cout << "Function value = " << std::setprecision(4) << std::fixed << fp << std::endl;
                }
                of->calcGradient(image->getImage(),xi, i);

                for(int i=0; i < image->getImageCount(); i++)
                {
                        calculateSandY<<<numBlocksNN, threadsPerBlockNN>>>(d_y, d_s, image->getImage(), xi, p_old, xi_old, (this->current_iteration-1)%this->K, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }

                LBFGS_recursion(d_y, d_s, xi, std::min(this->K,this->current_iteration), (this->current_iteration-1)%this->K, M, N);

                end = omp_get_wtime();
                double wall_time = end-start;
                if(verbose_flag) {
                        std::cout << "Time: "<< std::setprecision(4) << wall_time << " seconds" << std::endl;
                }
        }
        std::cout << "Too many iterations in LBFGS" << std::endl;
        of->calcFunction(image->getImage());
        deallocateMemoryGpu();
        return;
};

__host__ void LBFGS::LBFGS_recursion(float *d_y, float *d_s, float *xi, int par_M, int lbfgs_it, int M, int N){
        float **alpha, *aux_vector;
        float *d_r, *d_q;
        float rho = 0.0f;
        float rho_den;
        float beta = 0.0f;
        float sy = 0.0f;
        float yy = 0.0f;
        float sy_yy = 0.0f;
        alpha = (float**)malloc(image->getImageCount()*sizeof(float*));
        for(int i=0; i<image->getImageCount(); i++) {
                alpha[i] = (float*)malloc(par_M*sizeof(float));
        }

        for(int i=0; i<image->getImageCount(); i++) {
                memset (alpha[i],0,par_M*sizeof(float));
        }


        checkCudaErrors(cudaMalloc((void**)&aux_vector, sizeof(float)*M*N));
        checkCudaErrors(cudaMalloc((void**)&d_q, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(float)*M*N*image->getImageCount()));

        checkCudaErrors(cudaMemset(aux_vector, 0, sizeof(float)*M*N));
        checkCudaErrors(cudaMemset(d_r, 0, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemcpy(d_q, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));



        for(int i=0; i < I->getImageCount(); i++)
        {

                for(int k = par_M - 1; k >= 0; k--) {
                        //Rho_k = 1.0/(y_k's_k);
                        //printf("Image %d, Iter :%d\n", i, k);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                        rho_den = deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                        if(rho_den != 0.0f)
                                rho = 1.0/rho_den;
                        else
                                rho = 0.0f;
                        //printf("1. rho %f\n", rho);
                        //alpha_k = Rho_k x (s_k' * q);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_s, d_q, k, 0, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                        alpha[i][k] = rho * deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                        //printf("1. alpha %f\n", alpha[i][k]);
                        //q = q - alpha_k * y_k;
                        updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, -alpha[i][k], d_y, k, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }
        }

        //y_0'y_0
        //(s_0'y_0)/(y_0'y_0)
        for(int i=0; i < I->getImageCount(); i++)
        {
                getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, lbfgs_it, lbfgs_it, M, N, i);
                checkCudaErrors(cudaDeviceSynchronize());
                sy = deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);

                getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_y, lbfgs_it, lbfgs_it, M, N, i);
                checkCudaErrors(cudaDeviceSynchronize());
                yy = deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);

                if(yy!=0.0f)
                        sy_yy += sy/yy;
                else
                        sy_yy += 0.0f;

                //printf("sy: %f, yy: %f, sy_yy: %f\n",sy, yy, sy_yy);
        }




        for(int i=0; i < I->getImageCount(); i++)
        {
                // r = q x ((s_0'y_0)/(y_0'y_0));
                getR<<<numBlocksNN, threadsPerBlockNN>>>(d_r, d_q, sy_yy, M, N, i);
                checkCudaErrors(cudaDeviceSynchronize());

                for (int k = 0; k < par_M; k++) {
                        //Rho_k = 1.0/(y_k's_k);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                        //Calculate rho
                        rho_den = deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                        if(rho_den != 0.0f)
                                rho = 1.0f/rho_den;
                        else
                                rho = 0.0f;
                        //beta = rho * y_k' * r;
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_r, k, 0, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                        beta = rho * deviceReduce<float>(aux_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                        //printf("2. image %d - iter %d - rho: %f, alpha: %f, beta: %f, alpha-beta: %f\n", i, k, rho, alpha[i][k], beta, alpha[i][k]-beta);
                        //r = r + s_k * (alpha_k - beta)
                        updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_r, alpha[i][k]-beta, d_s, k, M, N, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }
        }

        //Changing the sign to -d_r
        for(int i=0; i < image->getImageCount(); i++)
        {
                searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(d_r, M, N, i); //Search direction
                checkCudaErrors(cudaDeviceSynchronize());
        }

        checkCudaErrors(cudaMemcpy(xi, d_r, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
        cudaFree(aux_vector);
        cudaFree(d_q);
        cudaFree(d_r);
        for(int i=0; i<image->getImageCount(); i++) {
                free(alpha[i]);
        }
        free(alpha);
};

namespace {
Optimizer *CreateLbfgs()
{
        return new LBFGS;
};

const std::string name = "CG-LBFGS";
const bool RegisteredLbgs = registerCreationFunction<Optimizer, std::string>(name, CreateLbfgs);
};
