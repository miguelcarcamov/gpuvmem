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

#include <cctype>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "cli/gpuvmem_cli_config.hh"
#include "utils/direction_cosines.cuh"
#include "utils/fixed_point.cuh"
#include "main.cuh"
#include "framework.cuh"
#include "kernels/gaussian2D.cuh"
#include "kernels/gaussianSinc2D.cuh"
#include "kernels/pillBox2D.cuh"
#include "kernels/pswf_12D.cuh"
#include "kernels/sinc2D.cuh"
#include "uvtaper.cuh"
#include "optimizers/conjugategradient.cuh"
#include "optimizers/lbfgs.cuh"
#include "linesearch/linesearcher.cuh"

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
  std::cout << "gpuvmem - GPU-accelerated radio synthesis imaging (MEM / RML framework)\n"
            << "Copyright (C) 2016-2020  Miguel Carcamo, Pablo Roman, Simon Casassus, Victor Moral, "
               "Fernando Rannou, Nicolás Muñoz\n"
            << "Contact: miguel.carcamo@protonmail.com\n"
            << "This program comes with ABSOLUTELY NO WARRANTY; for details use option -w\n"
            << "This is free software; use option -c for redistribution terms.\n\n";

  // Help does not require CUDA; print_help exits the process.
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
      print_help((argc > 0 && argv[0] != nullptr) ? argv[0] : nullptr);
    }
  }

  GpuvmemCliConfig cli{};
  if (!parse_gpuvmem_cli(argc, argv, cli, std::cerr)) {
    print_help((argc > 0 && argv[0] != nullptr) ? argv[0] : nullptr);
  }
  if (cli.runtime.print_warranty) {
    print_warranty();
    return 0;
  }
  if (cli.runtime.print_copyright) {
    print_copyright();
    return 0;
  }

  ////CHECK FOR AVAILABLE GPUs
  cudaError_t err = cudaGetDeviceCount(&num_gpus);

  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (code: "
              << static_cast<int>(err) << ")\n";
    std::cerr << "This usually means:\n";
    std::cerr << "  1. CUDA driver/runtime version mismatch\n";
    std::cerr << "  2. CUDA libraries not found (check LD_LIBRARY_PATH)\n";
    std::cerr << "  3. NVIDIA driver not properly installed\n";
    std::cerr
        << "  4. GPU not accessible (check permissions, CUDA_VISIBLE_DEVICES)\n";
    std::cerr << "\nTroubleshooting:\n";
    std::cerr << "  - Run: nvidia-smi (should show your GPU)\n";
    std::cerr
        << "  - Check: echo $LD_LIBRARY_PATH (should include CUDA lib path)\n";
    std::cerr << "  - Check: ls -la /opt/cuda/lib64/libcudart.so.*\n";
    return 1;
  }

  if (num_gpus < 1) {
    std::cerr << "No CUDA capable devices were detected\n";
    std::cerr << "This could mean:\n";
    std::cerr << "  - No GPUs are available\n";
    std::cerr << "  - GPUs are not CUDA-capable\n";
    std::cerr << "  - CUDA_VISIBLE_DEVICES is set incorrectly\n";
    return 1;
  }

  if (!IsAppBuiltAs64()) {
    std::cerr << argv[0]
              << " is only supported with on 64-bit OSs and the application must "
                 "be built as a 64-bit target. Test is being waived.\n";
    exit(EXIT_SUCCESS);
  }

  Synthesizer* sy = createObject<Synthesizer, std::string>("MFS");

  auto normalizeWeightingId = [](const std::string& in) -> std::string {
    std::string w = in;
    for (auto& c : w) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (w == "natural") return "Natural";
    if (w == "uniform") return "Uniform";
    if (w == "radial") return "Radial";
    if (w == "briggs" || w == "robust") return "Briggs";
    return in;
  };

  WeightingScheme* scheme = createObject<WeightingScheme, std::string>(
      normalizeWeightingId(cli.vars.weighting_scheme));

  Optimizer* cg =
      createObject<Optimizer, std::string>(cli.vars.optimizer_name);
  cg->setK(cli.vars.lbfgs_corrections);

  std::string linesearch_id = cli.vars.linesearch_name;
  const std::string& seeder_id = cli.vars.seeder_name;
  if (linesearch_id.empty() && !seeder_id.empty()) linesearch_id = "Brent";
  if (!linesearch_id.empty()) {
    LineSearcher* ls_raw =
        createObject<LineSearcher, std::string>(linesearch_id);
    std::unique_ptr<LineSearcher> ls_ptr(ls_raw);
    if (!seeder_id.empty()) {
      StepSizeSeeder* se_raw =
          createObject<StepSizeSeeder, std::string>(seeder_id);
      ls_ptr->setStepSizeSeeder(std::unique_ptr<StepSizeSeeder>(se_raw));
    }
    cg->setLineSearcher(std::move(ls_ptr));
  }

  // Antialiasing / gridding kernel (see createObject<CKernel,...> for named kernels).
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

  sy->setIoVisibilitiesHandler(ioms);
  sy->setIoImageHandler(iofits);
  sy->setWeightingScheme(scheme);
  sy->setGriddingKernel(sc);
  sy->setOptimizator(cg);
  sy->configure(cli);
  cg->setObjectiveFunction(of);

  // Filter *g = Singleton<FilterFactory>::Instance().CreateFilter(Gridding);
  // sy->applyFilter(g); // delete this line for no gridding

  sy->setDevice();  // Allocates penalizators[] from -Z (see MFS::setDevice)

  Image* syn_image = sy->getImage();
  of->setGridDimensions(syn_image->getN(), syn_image->getM(), syn_image->getImageCount());
  of->setRegularizationWeights(penalizators, nPenalizators);

  Fi* chi2 = createObject<Fi, std::string>("Chi2");
  Fi* e = createObject<Fi, std::string>("Entropy");
  Fi* l1 = createObject<Fi, std::string>("L1-Norm");
  Fi* tsqv = createObject<Fi, std::string>("TotalSquaredVariation");
  /*Fi* lap = createObject<Fi, std::string>("Laplacian");
  Fi* atv = createObject<Fi, std::string>("AnisotropicTotalVariation");
  Fi* itv = createObject<Fi, std::string>("IsotropicTotalVariation");*/

  // -Z list (penalizators[]): fewer than 5 floats: Chi2 uses fixed λ=1; weights map to
  // Entropy, L1, TSV in order (entries beyond the third are ignored; L2ConstantPrior is
  // not registered in main).
  // 5+ floats: Chi2, Entropy, L1, TSV, … (first entry scales χ²; extra slots ignored).
  // ObjectiveFunction::addFi omits terms whose λ is exactly 0 (see objectivefunction.cuh).
  const bool z_lists_chi2_weight =
      (penalizators != nullptr && nPenalizators >= 5);

  chi2->attachToObjectiveFunction(of);
  if (z_lists_chi2_weight) {
    chi2->configure(0, 0, 0, variables.normalize);
  } else {
    chi2->configure(-1, 0, 0, variables.normalize);
  }
  const int z_off = z_lists_chi2_weight ? 1 : 0;

  if (chi2->getPenalizationFactor() == 0.0f) {
    std::cerr
        << "WARNING: Chi-squared (data fidelity) weight is zero; that term is "
           "omitted from the objective (ObjectiveFunction::addFi skips λ=0).\n";
    if (z_lists_chi2_weight)
      std::cerr << "         With five or more -Z values, the first entry is "
                   "the Chi2 weight.\n";
  }

  e->attachToObjectiveFunction(of);
  e->configure(0 + z_off, 0, 0, false);
  e->setPrior(0.001f);
  l1->attachToObjectiveFunction(of);
  l1->configure(1 + z_off, 0, 0, false);
  tsqv->attachToObjectiveFunction(of);
  tsqv->configure(2 + z_off, 1, 1, false);

  // Short -Z lists: Fi::configure leaves TSV λ=0 when its index is past the list end.
  // Legacy default: if -Z is absent or too short for the TSV slot, use λ=0.05 for TSV.
  if (penalizators != nullptr) {
    const int tsqv_idx = 2 + z_off;
    if (tsqv_idx >= nPenalizators)
      tsqv->setPenalizationFactor(0.05f);
  } else {
    tsqv->setPenalizationFactor(0.05f);
  }

  // e->setPenalizationFactor(0.01); // If not used -Z (Fi::configure(-1,x,x))
  of->addFi(chi2);
  of->addFi(e);
  of->addFi(l1);
  of->addFi(tsqv);
  // sy->getImage()->getFunctionMapping()[i].evaluateXt = defaultEvaluateXt;
  // sy->getImage()->getFunctionMapping()[i].newP = defaultNewP;
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
