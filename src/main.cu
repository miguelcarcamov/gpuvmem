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
#include "gaussianSinc2D.cuh"
#include "gaussian2D.cuh"
#include "sinc2D.cuh"
#include "pswf_12D.cuh"
#include "fixedpoint.cuh"
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

/*
   This is a function that runs gpuvmem and calculates new regularization values according to the Belge et al. 2002 paper.
 */
std::vector<float> runGpuvmem(std::vector<float> args, Synthesizer *synthesizer)
{

        int cter = 0;
        std::vector<Fi*> fis = synthesizer->getOptimizator()->getObjectiveFuntion()->getFi();
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                if(cter)
                        (*it)->setPenalizationFactor(args[cter]);
                cter++;
        }

        synthesizer->clearRun();
        synthesizer->run();
        std::vector<float> fi_values = synthesizer->getOptimizator()->getObjectiveFuntion()->get_fi_values();
        std::vector<float> lambdas(fi_values.size(), 1.0f);

        for(int i=0; i < fi_values.size(); i++)
        {
                if(i>0)
                {
                        lambdas[i] = fi_values[0]/fi_values[i] * (logf(fi_values[i])/logf(fi_values[0]));
                        if(lambdas[i] < 0.0f)
                                lambdas[i] = 0.0f;
                }
        }

        return lambdas;
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
        enum {natural, uniform, briggs, radial}; // Weighting Schemes

        Synthesizer * sy = Singleton<SynthesizerFactory>::Instance().CreateSynthesizer(MFS);
        Optimizator * cg = Singleton<OptimizatorFactory>::Instance().CreateOptimizator(CG);
        // Choose your antialiasing kernel!
        CKernel * sc = new PillBox2D();
        //CKernel *sc = new Gaussian2D(7,7);
        //CKernel *sc = new Sinc2D(7,7);
        //CKernel * sc = new GaussianSinc2D(7, 7);
        //CKernel * sc = new PSWF_12D(7,7);
        //sc->setW1(2.50f);
        //CKernel * sc = Singleton<CKernelFactory>::Instance().CreateCKernel(gaussianSinc2D);
        ObjectiveFunction *of = Singleton<ObjectiveFunctionFactory>::Instance().CreateObjectiveFunction(DefaultObjectiveFunction);
        Io *ioms = Singleton<IoFactory>::Instance().CreateIo(MS); // This is the default Io Class
        WeightingScheme *scheme = Singleton<WeightingSchemeFactory>::Instance().CreateWeightingScheme(briggs);
        sy->setIoHandler(ioms);
        sy->setOrder(&optimizationOrder);
        sy->setWeightingScheme(scheme);
        sy->setGriddingKernel(sc);
        sy->configure(argc, argv);
        cg->setObjectiveFunction(of);
        sy->setOptimizator(cg);

        //Filter *g = Singleton<FilterFactory>::Instance().CreateFilter(Gridding);
        //sy->applyFilter(g); // delete this line for no gridding

        sy->setDevice(); // This routine sends the data to GPU memory
        Fi *chi2 = Singleton<FiFactory>::Instance().CreateFi(Chi2);
        Fi *e = Singleton<FiFactory>::Instance().CreateFi(Entropy);
        Fi *l1 = Singleton<FiFactory>::Instance().CreateFi(L1Norm);
        Fi *tsqv = Singleton<FiFactory>::Instance().CreateFi(TotalSquaredVariation);
        Fi *lap = Singleton<FiFactory>::Instance().CreateFi(Laplacian;
        chi2->configure(-1, 0, 0); // (penalizatorIndex, ImageIndex, imageToaddDphi)
        e->configure(0, 0, 0);
        l1->configure(1, 0, 0);
        tsqv->configure(2, 0, 0);
        lap->configure(3, 0, 0);
        //e->setPenalizationFactor(0.01); // If not used -Z (Fi.configure(-1,x,x))
        of->addFi(chi2);
        of->addFi(e);
        of->addFi(l1);
        of->addFi(tsqv);
        of->addFi(lap);
        //sy->getImage()->getFunctionMapping()[i].evaluateXt = particularEvaluateXt;
        //sy->getImage()->getFunctionMapping()[i].newP = particularNewP;
        //if the nopositivity flag is on  all images will run with no posivity,
        //otherwise the first image image will be calculated with positivity and all the others without positivity,
        //to modify this, use these sentences, where i corresponds to the index of the image ( particularly, means positivity)

        /*std::vector<float> lambdas = {1.0, 1e-5, 1e-5};
        std::vector<Fi*> fis = of->getFi();
        int i = 0;
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->setPenalizationFactor(lambdas[i]);
                i++;
        }

        std::vector<float> final_lambdas = fixedPointOpt(lambdas, &runGpuvmem, 1e-6, 60, sy);*/
        sy->run();

        sy->writeImages();
        sy->writeResiduals();
        sy->unSetDevice(); // This routine performs memory cleanup and release

        return 0;
}
