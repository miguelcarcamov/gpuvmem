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

#include "directioncosines.cuh"
#include "pillBox2D.cuh"
#include "pswf_12D.cuh"
#include "gaussianSinc2D.cuh"
#include "gaussian2D.cuh"
#include "sinc2D.cuh"
#include "pswf_12D.cuh"
#include "fixedpoint.cuh"
#include "uvtaper.cuh"
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
        std::vector<Fi*> fis = synthesizer->getOptimizator()->getObjectiveFunction()->getFi();
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                if(cter)
                        (*it)->setPenalizationFactor(args[cter]);
                cter++;
        }

        synthesizer->clearRun();
        synthesizer->run();
        std::vector<float> fi_values = synthesizer->getOptimizator()->getObjectiveFunction()->get_fi_values();
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

void optimizationOrder(Optimizer *optimizer, Image *image){
        optimizer->setImage(image);
        optimizer->setFlag(0);
        optimizer->optimize();
        /*optimizer->setFlag(1);
           optimizer->optimize();
           optimizer->setFlag(2);
           optimizer->optimize();
           optimizer->setFlag(3);
           optimizer->optimize();*/
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

        Synthesizer *sy = createObject<Synthesizer, std::string>("MFS");
        Optimizer *cg = createObject<Optimizer, std::string>("CG-FRPRMN");
        //Optimizer * cg = createObject<Optimizer, std::string>("CG-LBFGS");
        //cg->setK(15);
        // Choose your antialiasing kernel!
        CKernel *sc = new PillBox2D();
        //CKernel *sc = new Gaussian2D(7,7);
        //CKernel *sc = new Sinc2D(7,7);
        //CKernel *sc = new GaussianSinc2D(7, 7);
        //CKernel *sc = new PSWF_12D(9,9);
        //CKernel *sc = createObject<CKernel, std::string>("GaussianSinc2D");
        ObjectiveFunction *of = createObject<ObjectiveFunction, std::string>("ObjectiveFunction");
        Io *ioms = createObject<Io, std::string>("IoMS"); // This is the default Io Class
        Io *iofits = createObject<Io, std::string>("IoFITS"); // This is the default Io Class

        //UVTaper *uvtaper = new UVTaper(200000.0f); // Initialize your uvtaper in units of lambda
        WeightingScheme *scheme = createObject<WeightingScheme, std::string>("Briggs");
        //scheme->setUVTaper(uvtaper);

        sy->setIoVisibilitiesHandler(ioms);
        sy->setIoImageHandler(iofits);
        sy->setOrder(&optimizationOrder);
        sy->setWeightingScheme(scheme);
        sy->setGriddingKernel(sc);
        sy->setOptimizator(cg);
        sy->configure(argc, argv);
        cg->setObjectiveFunction(of);

        //Filter *g = Singleton<FilterFactory>::Instance().CreateFilter(Gridding);
        //sy->applyFilter(g); // delete this line for no gridding

        sy->setDevice(); // This routine sends the data to GPU memory
        Fi *chi2 = createObject<Fi,std::string>("Chi2");
        Fi *e = createObject<Fi,std::string>("Entropy");
        Fi *l1 = createObject<Fi,std::string>("L1-Norm");
        Fi *tsqv = createObject<Fi,std::string>("TotalSquaredVariation");
        Fi *lap = createObject<Fi,std::string>("Laplacian");

        chi2->configure(-1, 0, 0); // (penalizatorIndex, ImageIndex, imageToaddDphi)
        e->configure(0, 0, 0);
        e->setPrior(0.001f);
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
