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

#include "frprmn.cuh"
#include "directioncosines.cuh"
#include "pillBox2D.cuh"
#include "ellipticalGaussian2D.cuh"
#include "gaussianSinc2D.cuh"
#include "gaussian2D.cuh"
#include "sinc2D.cuh"
#include "pswf_02D.cuh"
#include "pswf_12D.cuh"
#include <time.h>

int num_gpus;

inline bool IsAppBuiltAs64()
{
  #if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
        return 1;
  #else
        return 0;
  #endif
}

void optimizationOrder(Optimizator *optimizator, Image *image){
        optimizator->setImage(image);
        optimizator->setFlag(0);
        optimizator->optimize();
        /*optimizator->setFlag(1);
        optimizator->optimize();
        optimizator->setFlag(2);
        optimizator->optimize();
        optimizator->setFlag(3);
        optimizator->optimize();*/
}

__host__ int main(int argc, char **argv) {
        ////CHECK FOR AVAILABLE GPUs
        cudaGetDeviceCount(&num_gpus);

        printf("gpuvmem Copyright (C) 2016-2020  Miguel Carcamo, Pablo Roman, Simon Casassus, Victor Moral, Fernando Rannou, Nicolás Muñoz - miguel.carcamo@protonmail.com\n");
        printf("This program comes with ABSOLUTELY NO WARRANTY; for details use option -w\n");
        printf("This is free software, and you are welcome to redistribute it under certain conditions; use option -c for details.\n\n\n");


        if(num_gpus < 1) {
                printf("No CUDA capable devices were detected\n");
                return 1;
        }

        if (!IsAppBuiltAs64()) {
                printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
                exit(EXIT_SUCCESS);
        }

        //// AVAILABLE CLASSES
        enum {MFS}; // Synthesizer
        enum {Chi2, Entropy, Laplacian, QuadraticPenalization, TotalVariation, TotalSquaredVariation, L1Norm}; // Fi
        enum {Gridding}; // Filter
        enum {CG, LBFGS}; // Optimizator
        enum {DefaultObjectiveFunction}; // ObjectiveFunction
        enum {MS}; // Io
        enum {SecondDerivative}; // Error calculation
        enum {pillbox2D, ellipticalGaussian2D, gaussian2D, sinc2D, gaussianSinc2D, pswf_02D, pswf_12D}; // CKernels for gridding

        Synthesizer * sy = Singleton<SynthesizerFactory>::Instance().CreateSynthesizer(MFS);
        Optimizator * cg = Singleton<OptimizatorFactory>::Instance().CreateOptimizator(CG);
        // Choose your antialiasing kernel!
        CKernel * sc = new PillBox2D(1,1);
        //CKernel * sc = new GaussianSinc2D(7, 7);
        //CKernel * sc = new PSWF_12D(3,3);
        //sc->setW1(2.50f);
        //CKernel * sc = Singleton<CKernelFactory>::Instance().CreateCKernel(gaussianSinc2D);
        ObjectiveFunction *of = Singleton<ObjectiveFunctionFactory>::Instance().CreateObjectiveFunction(DefaultObjectiveFunction);
        Io *ioms = Singleton<IoFactory>::Instance().CreateIo(MS); // This is the default Io Class
        sy->setIoHandler(ioms);
        sy->setOrder(&optimizationOrder);
        sy->setGriddingKernel(sc);
        sy->configure(argc, argv);
        cg->setObjectiveFunction(of);
        sy->setOptimizator(cg);

        //Filter *g = Singleton<FilterFactory>::Instance().CreateFilter(Gridding);
        //sy->applyFilter(g); // delete this line for no gridding

        sy->setDevice(); // This routine sends the data to GPU memory
        Fi *chi2 = Singleton<FiFactory>::Instance().CreateFi(Chi2);
        Fi *e = Singleton<FiFactory>::Instance().CreateFi(Entropy);
        Fi *l = Singleton<FiFactory>::Instance().CreateFi(Laplacian);
        chi2->configure(-1, 0, 0); // (penalizatorIndex, ImageIndex, imageToaddDphi)
        e->configure(0, 0, 0);
        l->configure(1, 0, 0);
        //e->setPenalizationFactor(0.01); // If not used -Z (Fi.configure(-1,x,x))
        of->addFi(chi2);
        of->addFi(e);
        of->addFi(l);
        //sy->getImage()->getFunctionMapping()[i].evaluateXt = particularEvaluateXt;
        //sy->getImage()->getFunctionMapping()[i].newP = particularNewP;
        //if the nopositivity flag is on  all images will run with no posivity,
        //otherwise the first image image will be calculated with positivity and all the others without positivity,
        //to modify this, use these sentences, where i corresponds to the index of the image ( particularly, means positivity)
        sy->run();
        std::vector<float> fi_values = of->get_fi_values();
        //for(int it = 0; it < fi_values.size() ; it++)
        //{
        //        std::cout << fi_values[it] << std::endl;
        //}

        //e->setPenalizationFactor(0.00001);
        //l->setPenalizationFactor(0.00005);
        //std::cout << "Trying to run another iteration" << std::endl;
        //sy->run();
        sy->unSetDevice(); // This routine performs memory cleanup and release

        return 0;
}
