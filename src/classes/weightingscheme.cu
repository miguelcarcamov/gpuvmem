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

#include "classes/weightingscheme.cuh"
#include <omp.h>
#include <iostream>

WeightingScheme::WeightingScheme() {
  this->threads = omp_get_num_procs() - 2;
  this->uvtaper = NULL;
  this->modify_weights = false;
}

WeightingScheme::WeightingScheme(int threads) {
  this->threads = threads;
  this->uvtaper = NULL;
  this->modify_weights = false;
}

WeightingScheme::WeightingScheme(int threads, UVTaper* uvtaper) {
  this->threads = threads;
  this->uvtaper = uvtaper;
  this->modify_weights = false;
}

WeightingScheme::WeightingScheme(int threads, UVTaper* uvtaper, bool modify_weights) {
  this->threads = threads;
  this->uvtaper = uvtaper;
  this->modify_weights = modify_weights;
}

bool WeightingScheme::getModifyWeights() {
  return this->modify_weights;
}

void WeightingScheme::setModifyWeights(bool modify_weights) {
  this->modify_weights = modify_weights;
}

int WeightingScheme::getThreads() {
  return this->threads;
}

void WeightingScheme::setThreads(int threads) {
  this->threads = threads;
  std::cout << "The running weighting scheme threads have been set to "
            << this->threads << std::endl;
}

UVTaper* WeightingScheme::getUVTaper() {
  return this->uvtaper;
}

void WeightingScheme::setUVTaper(UVTaper* uvtaper) {
  this->uvtaper = uvtaper;
  std::cout << "UVTaper has been set" << std::endl;
  std::cout << "UVTaper Features - bmaj=" << this->uvtaper->getSigma_maj()
            << ", bmin=" << this->uvtaper->getSigma_min()
            << ", bpa=" << this->uvtaper->getBPA() << std::endl;
}

void WeightingScheme::restoreWeights(std::vector<MSDataset>& d) {
  for (int j = 0; j < d.size(); j++) {
    for (int f = 0; f < d[j].data.nfields; f++) {
      for (int i = 0; i < d[j].data.total_frequencies; i++) {
        for (int s = 0; s < d[j].data.nstokes; s++) {
          d[j].fields[f].visibilities[i][s].weight.assign(
              d[j].fields[f].backup_visibilities[i][s].weight.begin(),
              d[j].fields[f].backup_visibilities[i][s].weight.end());
        }
      }
    }
  }
}
