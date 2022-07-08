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

#include "linmin.cuh"
#define TOL 1.0e-7

float *device_pcom;
float *device_xicom, (*nrfunc)(float *);
extern long M;
extern long N;
extern float MINPIX, eta;
extern bool nopositivity;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int verbose_flag;
extern int image_count;
extern Image *I;

extern ObjectiveFunction *testof;

__host__ void linmin(float *p, float *xi, float *fret,
                     float (*func)(float *))  // p and xi are in GPU
{
  float xx, xmin, fx, fb, fa, bx, ax;

  checkCudaErrors(
      cudaMalloc((void **)&device_pcom, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMemset(device_pcom, 0, sizeof(float) * M * N * image_count));

  checkCudaErrors((
      cudaMalloc((void **)&device_xicom, sizeof(float) * M * N * image_count)));
  checkCudaErrors(
      cudaMemset(device_xicom, 0, sizeof(float) * M * N * image_count));
  nrfunc = func;
  // device_pcom = p;
  // device_xicom = xi;
  checkCudaErrors(cudaMemcpy(device_pcom, p,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(device_xicom, xi,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));

  ax = 0.0;
  xx = 1.0;

  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);

  *fret = brent(ax, xx, bx, TOL, &xmin, f1dim);
  if (verbose_flag) {
    printf("xmin = %f\n\n", xmin);
  }

  // GPU MUL AND ADD
  // xi     = xi*xmin;
  // p      = p + xi;
  imageMap *auxPtr = I->getFunctionMapping();
  if (!nopositivity) {
    for (int i = 0; i < I->getImageCount(); i++) {
      (auxPtr[i].newP)(p, xi, xmin, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    for (int i = 0; i < I->getImageCount(); i++) {
      newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M,
                                                           i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    /*evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(device_xt,
       device_pcom, device_xicom, x, N, M, 0);
       checkCudaErrors(cudaDeviceSynchronize());*/
  }

  cudaFree(device_xicom);
  cudaFree(device_pcom);
}
#undef TOL
