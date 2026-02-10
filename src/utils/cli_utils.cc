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

#include "utils/cli_utils.hh"
#include "flags.cuh"
#include <cstdlib>

// Extern variables
extern Flags flags;
extern bool verbose_flag, nopositivity, apply_noise, print_images, print_errors,
    save_model_input, radius_mask, modify_weights;

void print_help() {
  flags.PrintHelp();
}

Vars getOptions(int argc, char** argv) {
  Vars variables;
  bool help, copyright, warranty;

  flags.Var(variables.input, 'i', "input", std::string("NULL"),
            "Name of the input visibility file/s (separated by a comma)",
            "Mandatory");
  flags.Var(variables.output, 'o', "output", std::string("NULL"),
            "Name of the output visibility file/s (separated by a comma)",
            "Mandatory");
  flags.Var(variables.output_image, 'O', "output_image",
            std::string("mod_out.fits"),
            "Name of the output visibility file/s (separated by a comma)");
  flags.Var(variables.modin, 'm', "model_input", std::string("mod_in_0.fits"),
            "FITS file including a complete header for astrometry",
            "Mandatory");
  flags.Var(variables.noise, 'n', "noise", -1.0f, "Noise factor parameter",
            "Optional");
  flags.Var(
      variables.eta, 'e', "eta", -1.0f,
      "Variable that controls the minimum image value in the entropy prior");
  flags.Var(variables.noise_cut, 'N', "noise_cut", 10.0f, "Noise-cut Parameter",
            "Optional");
  flags.Var(variables.nu_0, 'F', "ref_frequency", -1.0f,
            "Reference frequency in Hz (if alpha is not zero). It will be "
            "calculated from the measurement set if not set",
            "Optional");
  flags.Var(variables.threshold, 'T', "threshold", 0.0f,
            "Threshold to calculate the spectral index image above a certain "
            "number of sigmas in I_nu_0");
  flags.Var(
      variables.alpha_n_sigma, 'A', "alpha_sigma_cut", 3.0f,
      "Mask alpha to 0 where I_nu_0 < alpha_sigma_cut * noise (e.g. 3 or 5). "
      "When > 0, overrides threshold for alpha masking.");
  flags.Var(variables.path, 'p', "path", std::string("mem/"),
            "Path to save FITS images. With last trail / included. (Example "
            "./../mem/)");
  flags.Var(variables.gpus, 'G', "gpus", std::string("0"),
            "Index of the GPU/s you are going to use separated by a comma");
  flags.Var(variables.randoms, 'r', "random_sampling", 1.0f,
            "Percentage of data used when random sampling", "Optional");
  flags.Var(
      variables.robust_param, 'R', "robust_parameter", 2.0f,
      "Robust weighting parameter when gridding. -2.0 for uniform weighting, "
      "2.0 for natural weighting and 0.0 for a tradeoff between these two.");
  flags.Var(variables.ofile, 'f', "output_file", std::string("NULL"),
            "Output file where final objective function values are saved",
            "Optional");
  flags.Var(variables.blockSizeX, 'X', "blockSizeX", int32_t(-1),
            "GPU block X Size for image/Fourier plane (Needs to be pow of 2)");
  flags.Var(variables.blockSizeY, 'Y', "blockSizeY", int32_t(-1),
            "GPU block Y Size for image/Fourier plane (Needs to be pow of 2)");
  flags.Var(variables.blockSizeV, 'V', "blockSizeV", int32_t(-1),
            "GPU block V Size for visibilities (Needs to be pow of 2)");
  flags.Var(variables.it_max, 't', "iterations", int32_t(500),
            "Number of iterations for optimization");
  flags.Var(variables.gridding, 'g', "gridding", int32_t(0),
            "Use gridded visibilities. This is done in CPU (Need to select the "
            "CPU threads that will grid the input visibilities)");
  flags.Var(variables.initial_values, 'z', "initial_values",
            std::string("NULL"), "Initial values for image/s");
  flags.Var(
      variables.penalization_factors, 'Z', "regularization_factors",
      std::string("NULL"),
      "Regularization factors for each regularization (separated by a comma)");
  flags.Var(variables.user_mask, 'U', "user-mask", std::string("NULL"),
            "Use a user created mask instead of using the noise mask");
  flags.Bool(verbose_flag, 'v', "verbose",
             "Shows information through all the execution", "Flags");
  flags.Bool(nopositivity, 'x', "nopositivity",
             "Runs gpuvmem with no positivity restrictions on the images",
             "Flags");
  flags.Bool(apply_noise, 'a', "apply-noise",
             "Applies random gaussian noise to visibilities", "Flags");
  flags.Bool(print_images, 'P', "print-images", "Prints images per iteration",
             "Flags");
  flags.Bool(print_errors, 'E', "print-errors", "Prints final error maps",
             "Flags");
  flags.Bool(save_model_input, 's', "save_modelcolumn",
             "Saves the model visibilities on the model column of the input MS",
             "Flags");
  flags.Bool(radius_mask, 'M', "use-radius-mask",
             "Use a mask based on a radius instead of the noise estimation",
             "Flags");
  flags.Bool(modify_weights, 'W', "modify-weights",
             "Modify Measurement Set WEIGHT column with gpuvmem weights",
             "Flags");
  flags.Bool(variables.normalize, 'l', "normalize",
             "Normalize chi-squared by effective number of samples", "Flags");
  flags.Var(variables.optimization_mode, 'J', "optimization-mode",
            std::string("alpha_static"),
            "Optimization strategy: joint (I_nu_0+alpha), block (alternate), one (single image), alpha_static (I_nu_0 only)");
  flags.Bool(help, 'h', "help", "Shows this help", "Help");
  flags.Bool(warranty, 'w', "warranty", "Shows warranty details", "Help");
  flags.Bool(copyright, 'c', "copyright", "Shows copyright conditions", "Help");

  if (!flags.Parse(argc, argv)) {
    print_help();
    exit(EXIT_SUCCESS);
  } else if (help) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (warranty) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (copyright) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (variables.randoms > 1.0 || variables.randoms < 0.0) {
    print_help();
    exit(EXIT_FAILURE);
  }

  if (variables.gridding < 0) {
    print_help();
    exit(EXIT_FAILURE);
  }

  if (variables.user_mask != "NULL") {
    variables.noise_cut = 1.0f;
  }

  return variables;
}
