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

#include "frprmn.cuh"

extern long M;
extern long N;

ObjectiveFunction *testof;
Image *I;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int verbose_flag;
int flag_opt;

#define EPS 1.0e-10

#define FREEALL cudaFree(device_gg_vector); cudaFree(device_dgg_vector); cudaFree(xi); cudaFree(device_h); cudaFree(device_g);

__host__ void ConjugateGradient::allocateMemoryGpu()
{
        checkCudaErrors(cudaMalloc((void**)&device_g, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(device_g, 0, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMalloc((void**)&device_h, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(device_h, 0, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMalloc((void**)&xi, sizeof(float)*M*N*image->getImageCount()));
        checkCudaErrors(cudaMemset(xi, 0, sizeof(float)*M*N*image->getImageCount()));

        checkCudaErrors(cudaMalloc((void**)&device_gg_vector, sizeof(float)*M*N));
        checkCudaErrors(cudaMemset(device_gg_vector, 0, sizeof(float)*M*N));

        checkCudaErrors(cudaMalloc((void**)&device_dgg_vector, sizeof(float)*M*N));
        checkCudaErrors(cudaMemset(device_dgg_vector, 0, sizeof(float)*M*N));
};
__host__ void ConjugateGradient::deallocateMemoryGpu()
{
        FREEALL
};

__host__ void ConjugateGradient::optimize()
{
        printf("\n\nStarting Fletcher Reeves Polak Ribiere method (Conj. Grad.)\n\n");
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
                printf("Starting function value = %f\n", fp);
        }
        of->calcGradient(image->getImage(),xi,0);
        //g=-xi
        //xi=h=g

        for(int i=0; i < image->getImageCount(); i++)
        {
                searchDirection<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, N, M, i); //Search direction
                checkCudaErrors(cudaDeviceSynchronize());
        }
        ////////////////////////////////////////////////////////////////
        for(int i=1; i <= this->total_iterations; i++) {
                start = omp_get_wtime();
                this->current_iteration = i;
                if(verbose_flag) {
                        printf("\n\n********** Iteration %d **********\n\n", i);
                }
                linmin(image->getImage(), xi, &fret, NULL);
                if (2.0f*fabsf(fret-fp) <= this->ftol*(fabsf(fret)+fabsf(fp)+EPS)) {
                        printf("Exit due to tolerance\n");
                        of->calcFunction(I->getImage());
                        deallocateMemoryGpu();
                        return;
                }

                fp= of->calcFunction(image->getImage());
                if(verbose_flag) {
                        printf("Function value = %f\n", fp);
                }
                of->calcGradient(image->getImage(),xi, i);
                dgg = gg = 0.0;
                ////gg = g*g
                ////dgg = (xi+g)*xi
                checkCudaErrors(cudaMemset(device_gg_vector, 0, sizeof(float)*M*N));
                checkCudaErrors(cudaMemset(device_dgg_vector, 0, sizeof(float)*M*N));
                for(int i=0; i < image->getImageCount(); i++)
                {
                        getGGandDGG<<<numBlocksNN, threadsPerBlockNN>>>(device_gg_vector, device_dgg_vector, xi, device_g, N, M, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }
                ////getSums (Reductions) of gg dgg
                gg = deviceReduce<float>(device_gg_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                dgg = deviceReduce<float>(device_dgg_vector, M*N, threadsPerBlockNN.x * threadsPerBlockNN.y);
                if(gg == 0.0) {
                        printf("Exit due to gg = 0\n");
                        of->calcFunction(image->getImage());
                        deallocateMemoryGpu();
                        return;
                }
                gam = fmax(0.0f, dgg/gg);
                //printf("Gamma = %f\n", gam);
                //g=-xi
                //xi=h=g+gam*h;
                for(int i=0; i < image->getImageCount(); i++)
                {
                        newXi<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, gam, N, M, i);
                        checkCudaErrors(cudaDeviceSynchronize());
                }
                end = omp_get_wtime();
                double wall_time = end-start;
                if(verbose_flag) {
                        printf("Time: %lf seconds\n", i, wall_time);
                }
        }
        printf("Too many iterations in frprmn\n");
        of->calcFunction(image->getImage());
        deallocateMemoryGpu();
        return;
};

namespace {
Optimizer *CreateFrprmn()
{
        return new ConjugateGradient;
};

const std::string name = "CG-FRPRMN";
const bool RegisteredFrprmn = registerCreationFunction<Optimizer, std::string>(name, CreateFrprmn);
};
