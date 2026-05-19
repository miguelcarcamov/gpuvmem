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

#include "utils/physics_utils.cuh"
#include <cmath>

__host__ __device__ float freq_to_wavelength(float freq) {
  float lambda = LIGHTSPEED / freq;
  return lambda;
}

__host__ __device__ double metres_to_lambda(double uvw_metres, float freq) {
  float lambda = freq_to_wavelength(freq);
  double uvw_lambda = uvw_metres / lambda;
  return uvw_lambda;
}

__host__ __device__ float distance(float x, float y, float x0, float y0) {
  float sumsqr = (x - x0) * (x - x0) + (y - y0) * (y - y0);
  float distance = sqrtf(sumsqr);
  return distance;
}
