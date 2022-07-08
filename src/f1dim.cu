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

#include "f1dim.cuh"
extern float* device_pcom;
extern float *device_xicom, (*nrfunc)(float*);
extern long M;
extern long N;
extern float MINPIX, eta;
extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern bool nopositivity;
extern float* initial_values;
extern int image_count;

extern ObjectiveFunction* testof;
extern Image* I;

__host__ float f1dim(float x) {
  float* device_xt;
  float f;

  checkCudaErrors(
      cudaMalloc((void**)&device_xt, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMemset(device_xt, 0, sizeof(float) * M * N * image_count));

  imageMap* auxPtr = I->getFunctionMapping();
  // xt = pcom+x*xicom;
  if (!nopositivity) {
    for (int i = 0; i < I->getImageCount(); i++) {
      (auxPtr[i].evaluateXt)(device_xt, device_pcom, device_xicom, x, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    for (int i = 0; i < I->getImageCount(); i++) {
      evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(
          device_xt, device_pcom, device_xicom, x, N, M, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    /*evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(device_xt,
       device_pcom, device_xicom, x, N, M, 0);
       checkCudaErrors(cudaDeviceSynchronize());*/
  }

  // f = (*nrfunc)(device_xt);
  f = testof->calcFunction(device_xt);
  cudaFree(device_xt);
  return f;
}
