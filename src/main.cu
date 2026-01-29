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

#include <time.h>

#include "directioncosines.cuh"
#include "fixedpoint.cuh"
#include "framework.cuh"
#include "gaussian2D.cuh"
#include "gaussianSinc2D.cuh"
#include "pillBox2D.cuh"
#include "pswf_12D.cuh"
#include "sinc2D.cuh"
#include "uvtaper.cuh"
#include "optimizers/conjugategradient.cuh"
#include "lbfgs.cuh"
#include "linesearcher.cuh"
#include <memory>

// Note: Optimizer factory registrations happen automatically when
// conjugategradient.cu and lbfgs.cu are compiled and linked.
// All available optimizers are registered via the factory pattern.
// Line searcher and seeder factory registrations happen automatically when
// their respective .cu files are compiled and linked.

extern Vars variables;

int num_gpus;

inline bool IsAppBuiltAs64() {
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
  return 1;
#else
  return 0;
#endif
}

/*
   This is a function that runs gpuvmem and calculates new regularization values
   according to the Belge et al. 2002 paper.
 */
std::vector<float> runGpuvmem(std::vector<float> args,
                              Synthesizer* synthesizer) {
  int cter = 0;
  std::vector<Fi*> fis =
      synthesizer->getOptimizator()->getObjectiveFunction()->getFi();
  for (std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++) {
    if (cter)
      (*it)->setPenalizationFactor(args[cter]);
    cter++;
  }

  synthesizer->clearRun();
  synthesizer->run();
  std::vector<float> fi_values =
      synthesizer->getOptimizator()->getObjectiveFunction()->get_fi_values();
  std::vector<float> lambdas(fi_values.size(), 1.0f);

  for (int i = 0; i < fi_values.size(); i++) {
    if (i > 0) {
      lambdas[i] = fi_values[0] / fi_values[i] *
                   (logf(fi_values[i]) / logf(fi_values[0]));
      if (lambdas[i] < 0.0f)
        lambdas[i] = 0.0f;
    }
  }

  return lambdas;
}

__host__ int main(int argc, char** argv) {
  ////CHECK FOR AVAILABLE GPUs
  cudaError_t err = cudaGetDeviceCount(&num_gpus);

  printf(
      "gpuvmem Copyright (C) 2016-2020  Miguel Carcamo, Pablo Roman, Simon "
      "Casassus, Victor Moral, Fernando Rannou, Nicolás Muñoz - "
      "miguel.carcamo@protonmail.com\n");
  printf(
      "This program comes with ABSOLUTELY NO WARRANTY; for details use option "
      "-w\n");
  printf(
      "This is free software, and you are welcome to redistribute it under "
      "certain conditions; use option -c for details.\n\n\n");

  if (err != cudaSuccess) {
    printf("CUDA Error: %s (code: %d)\n", cudaGetErrorString(err), err);
    printf("This usually means:\n");
    printf("  1. CUDA driver/runtime version mismatch\n");
    printf("  2. CUDA libraries not found (check LD_LIBRARY_PATH)\n");
    printf("  3. NVIDIA driver not properly installed\n");
    printf(
        "  4. GPU not accessible (check permissions, CUDA_VISIBLE_DEVICES)\n");
    printf("\nTroubleshooting:\n");
    printf("  - Run: nvidia-smi (should show your GPU)\n");
    printf("  - Check: echo $LD_LIBRARY_PATH (should include CUDA lib path)\n");
    printf("  - Check: ls -la /opt/cuda/lib64/libcudart.so.*\n");
    return 1;
  }

  if (num_gpus < 1) {
    printf("No CUDA capable devices were detected\n");
    printf("This could mean:\n");
    printf("  - No GPUs are available\n");
    printf("  - GPUs are not CUDA-capable\n");
    printf("  - CUDA_VISIBLE_DEVICES is set incorrectly\n");
    return 1;
  }

  if (!IsAppBuiltAs64()) {
    printf(
        "%s is only supported with on 64-bit OSs and the application must be "
        "built as a 64-bit target. Test is being waived.\n",
        argv[0]);
    exit(EXIT_SUCCESS);
  }

  Synthesizer* sy = createObject<Synthesizer, std::string>("MFS");
  
  // Choose your optimizer! Available options:
  // Conjugate Gradient variants:
  //   "CG-FletcherReeves"      - Classic FR method (always non-negative beta)
  //   "CG-PolakRibiere"         - PRP method (often faster for nonlinear problems)
  //   "CG-HestenesStiefel"      - HS method (theoretical foundation)
  //   "CG-LiuStorey"            - LS method (good for image restoration)
  //   "CG-DaiYuan"              - DY method (global convergence guarantees)
  //   "CG-HagerZhang"           - HZ method (excellent practical performance)
  //   "CG-RMIL"                 - RMIL method (good for image recovery)
  // Limited Memory BFGS:
  //   "CG-LBFGS"                - L-BFGS method (quasi-Newton, good for large problems)
  
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-FletcherReeves");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-PolakRibiere");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-HestenesStiefel");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-LiuStorey");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-DaiYuan");
  Optimizer* cg = createObject<Optimizer, std::string>("CG-HagerZhang");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-RMIL");
  // Optimizer* cg = createObject<Optimizer, std::string>("CG-LBFGS");
  // cg->setK(10);
  // For LBFGS, you can set the memory limit (number of correction pairs):
  // if (cg->getK() > 0) {  // Check if it's LBFGS
  //   cg->setK(15);  // Use 15 correction pairs (default is 100)
  // }
  
  // ========================================================================
  // Configure Line Search and Seeder (Optional)
  // ========================================================================
  // By default, CG-HagerZhang uses Brent line search. You can customize it:
  //
  // Example: Use GLL Armijo with BB Min1 seeder (recommended for faster convergence)
  // Step 1: Create LineSearcher using factory (returns raw pointer)
  /*LineSearcher* searcher_ptr = createObject<LineSearcher, std::string>("GLLArmijo");
  // Step 2: Create StepSizeSeeder using factory (returns raw pointer)
  StepSizeSeeder* seeder_ptr = createObject<StepSizeSeeder, std::string>("BBMin1Seeder");
  // Step 3: Wrap in unique_ptr and assign seeder to line searcher
  auto searcher = std::unique_ptr<LineSearcher>(searcher_ptr);
  searcher->setStepSizeSeeder(std::unique_ptr<StepSizeSeeder>(seeder_ptr));
  
  // Step 4: Optional - Set initial step size on line searcher (default is 1.0)
  // Use a larger value (e.g., 2.0, 5.0, 10.0) if you want to start with bigger steps
  searcher->setInitialStepSize(100.0f);  // Change this value to start with a bigger alpha
  
  // Step 5: Assign line searcher to optimizer (can call directly on Optimizer*)
  cg->setLineSearcher(std::move(searcher));*/
  //
  // Other options:
  //   Line searchers: "GLLArmijo", "BacktrackingArmijo", "FistaBacktracking", "Brent", "GoldenSectionSearch", "Fixed"
  //   Seeders: "BBMin1Seeder", "BBMin2Seeder", "BBAlternatingSeeder", "CubicInterpolationSeeder", "QuadraticInterpolationSeeder"
  //   (Seeders are optional - set on the line searcher before passing to optimizer)
  //  Choose your antialiasing kernel!
  CKernel* sc = new PillBox2D();
  // CKernel *sc = new Gaussian2D(7,7);
  // CKernel *sc = new Sinc2D(7,7);
  // CKernel *sc = new GaussianSinc2D(7, 7);
  // CKernel *sc = new PSWF_12D(9,9);
  // CKernel *sc = createObject<CKernel, std::string>("GaussianSinc2D");
  ObjectiveFunction* of =
      createObject<ObjectiveFunction, std::string>("ObjectiveFunction");
  Io* ioms =
      createObject<Io, std::string>("IoMS");  // This is the default Io Class
  Io* iofits =
      createObject<Io, std::string>("IoFITS");  // This is the default Io Class

  // UVTaper *uvtaper = new UVTaper(200000.0f); // Initialize your uvtaper in
  // units of lambda
  WeightingScheme* scheme =
      createObject<WeightingScheme, std::string>("Natural");
  // scheme->setUVTaper(uvtaper);

  sy->setIoVisibilitiesHandler(ioms);
  sy->setIoImageHandler(iofits);
  sy->setWeightingScheme(scheme);
  sy->setGriddingKernel(sc);
  sy->setOptimizator(cg);
  sy->configure(argc, argv);
  cg->setObjectiveFunction(of);

  // Filter *g = Singleton<FilterFactory>::Instance().CreateFilter(Gridding);
  // sy->applyFilter(g); // delete this line for no gridding

  sy->setDevice();  // This routine sends the data to GPU memory
  Fi* chi2 = createObject<Fi, std::string>("Chi2");
  Fi* e = createObject<Fi, std::string>("Entropy");
  Fi* l1 = createObject<Fi, std::string>("L1-Norm");
  Fi* tsqv = createObject<Fi, std::string>("TotalSquaredVariation");
  Fi* l2cp = createObject<Fi, std::string>("L2ConstantPrior");
  /*Fi* lap = createObject<Fi, std::string>("Laplacian");
  Fi* atv = createObject<Fi, std::string>("AnisotropicTotalVariation");
  Fi* itv = createObject<Fi, std::string>("IsotropicTotalVariation");*/

  chi2->configure(-1, 0, 0,
                  variables.normalize);  // (penalizatorIndex, ImageIndex,
                                         // imageToaddDphi, normalize)
  e->configure(0, 0, 0, false);
  e->setPrior(0.001f);
  l1->configure(1, 0, 0, false);
  tsqv->configure(2, 1, 1, false);
  tsqv->setPenalizationFactor(0.05f);
  l2cp->configure(3, 1, 1, false);
  l2cp->setPenalizationFactor(0.05f);
  l2cp->setPrior(2.0f);
  // e->setPenalizationFactor(0.01); // If not used -Z (Fi.configure(-1,x,x))
  of->addFi(chi2);
  of->addFi(e);
  of->addFi(l1);
  of->addFi(tsqv);
  of->addFi(l2cp);
  // sy->getImage()->getFunctionMapping()[i].evaluateXt = particularEvaluateXt;
  // sy->getImage()->getFunctionMapping()[i].newP = particularNewP;
  // if the nopositivity flag is on  all images will run with no posivity,
  // otherwise the first image image will be calculated with positivity and all
  // the others without positivity, to modify this, use these sentences, where i
  // corresponds to the index of the image ( particularly, means positivity)

  /*std::vector<float> lambdas = {1.0, 1e-5, 1e-5};
     std::vector<Fi*> fis = of->getFi();
     int i = 0;
     for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
     {
          (*it)->setPenalizationFactor(lambdas[i]);
          i++;
     }

     std::vector<float> final_lambdas = fixedPointOpt(lambdas, &runGpuvmem,
     1e-6, 60, sy);*/
  sy->run();

  sy->writeImages();
  sy->writeResiduals();
  sy->unSetDevice();  // This routine performs memory cleanup and release

  return 0;
}
