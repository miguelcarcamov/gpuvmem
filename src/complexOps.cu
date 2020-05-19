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

#include "complexOps.cuh"


__host__ __device__ cufftComplex multRealComplex(cufftComplex c1, float c2)
{

        cufftComplex result;

        result.x = c1.x * c2;
        result.y = c1.y * c2;

        return result;

}
__host__ __device__ cufftComplex multComplexComplex(cufftComplex c1, cufftComplex c2)
{

        cufftComplex result;

        result.x = (c1.x * c2.x) - (c1.y * c2.y);
        result.y = (c1.x * c2.y) + (c1.y * c2.x);
        return result;

}

__host__ __device__ cufftComplex divComplexComplex(cufftComplex c1, cufftComplex c2)
{

        cufftComplex result;
        float r, den;

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

__host__ __device__ cufftComplex addComplexComplex(cufftComplex c1, cufftComplex c2)
{

        cufftComplex result;

        result.x = c1.x + c2.x;
        result.y = c1.y + c2.y;
        return result;

}

__host__ __device__ cufftComplex subComplexComplex(cufftComplex c1, cufftComplex c2)
{

        cufftComplex result;

        result.x = c1.x - c2.x;
        result.y = c1.y - c2.y;
        return result;

}

__host__ __device__ cufftComplex ConjComplex(cufftComplex c1)
{

        cufftComplex result;

        result.x = c1.x;
        result.y = -c1.y;
        return result;

}
