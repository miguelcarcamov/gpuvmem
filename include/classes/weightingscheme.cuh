#ifndef WEIGHTINGSCHEME_CUH
#define WEIGHTINGSCHEME_CUH

#include "MSFITSIO.cuh"
#include "uvtaper.cuh"
#include <vector>

class WeightingScheme {
public:
virtual void apply(std::vector<MSDataset>& d) = 0;
virtual void configure(void *params) = 0;

WeightingScheme(){
    this->threads = omp_get_num_procs() - 2;
    this->uvtaper = NULL;
    this->modify_weights = false;
};

WeightingScheme(int threads){
    this->threads = threads;
    this->uvtaper = NULL;
    this->modify_weights = false;
};

WeightingScheme(int threads, UVTaper *uvtaper){
    this->threads = threads;
    this->uvtaper = uvtaper;
    this->modify_weights = false;
};

WeightingScheme(int threads, UVTaper *uvtaper, bool modify_weights){
    this->threads = threads;
    this->uvtaper = uvtaper;
    this->modify_weights = modify_weights;
};


bool getModifyWeights(){
    return this->modify_weights;
};

void setModifyWeights(bool modify_weights){
    this->modify_weights = modify_weights;
};

int getThreads(){
    return this->threads;
};

void setThreads(int threads){
    this->threads = threads;
    std::cout << "The running weighting scheme threads have been set to "<< this->threads << std::endl;
};

UVTaper * getUVTaper(){
    return this->uvtaper;
};

void setUVTaper(UVTaper * uvtaper){
    this->uvtaper = uvtaper;
    std::cout << "UVTaper has been set" << std::endl;
    std::cout << "UVTaper Features - bmaj=" << this->uvtaper->getSigma_maj() << ", bmin=" << this->uvtaper->getSigma_min() << ", bpa=" << this->uvtaper->getBPA() << std::endl;
};

void restoreWeights(std::vector<MSDataset>& d){
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        d[j].fields[f].visibilities[i][s].weight.assign(d[j].fields[f].backup_visibilities[i][s].weight.begin(), d[j].fields[f].backup_visibilities[i][s].weight.end());
                                }
                        }
                }
        }
};

protected:
  int threads;
  UVTaper *uvtaper = NULL;
  bool modify_weights;
};
#endif
