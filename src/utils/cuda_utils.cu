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

#include "utils/cuda_utils.cuh"
#include "utils/math_utils.hh"
#include <helper_cuda.h>
#include "error.cuh"
#include <cuda_runtime.h>
#include <iostream>

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

__host__ void getNumBlocksAndThreads(int n,
                                     int maxBlocks,
                                     int maxThreads,
                                     int& blocks,
                                     int& threads,
                                     bool reduction) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  if (reduction) {
    threads = (n < maxThreads * 2) ? NearestPowerOf2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  } else {
    threads = (n < maxThreads) ? NearestPowerOf2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  }
  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    std::cerr << "n is too large, please choose a smaller number!\n";
  }

  if (blocks > prop.maxGridSize[0]) {
    std::cerr << "Grid size <" << blocks << "> exceeds the device capability <"
              << prop.maxGridSize[0] << ">, set block size as " << (threads * 2)
              << " (original " << threads << ")\n";

    blocks /= 2;
    threads *= 2;
  }

  if (reduction) {
    blocks = MIN(maxBlocks, blocks);
  }
}
