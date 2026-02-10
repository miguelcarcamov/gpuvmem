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

#include "utils/error_handling.cuh"
#include "error.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Extern variables needed by goToError
extern int num_gpus, firstgpu;

__host__ void goToError() {
  if (num_gpus > 1) {
    for (int i = firstgpu + 1; i < firstgpu + num_gpus; i++) {
      cudaSetDevice(firstgpu);
      cudaDeviceDisablePeerAccess(i);
      cudaSetDevice(i);
      cudaDeviceDisablePeerAccess(firstgpu);
    }

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice((i % num_gpus) + firstgpu);
      cudaDeviceReset();
    }
  }

  printf("An error has ocurred, exiting\n");
  exit(0);
}
