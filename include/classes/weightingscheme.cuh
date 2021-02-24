#ifndef WEIGHTINGSCHEME_CUH
#define WEIGHTINGSCHEME_CUH

#include "MSFITSIO.cuh"
#include <vector>

class WeightingScheme {
public:
virtual void apply(std::vector<MSDataset>& d) = 0;
virtual void configure(void* params) = 0;

WeightingScheme(){
    this->threads = omp_get_max_threads() - 1;
};

WeightingScheme(int threads){
    this->threads = threads;
};

int getThreads(){
    return this->threads;
};

int setThreads(int threads){
    this->threads = threads;
    printf("The running weighting scheme threads has been set to %d\n", this->threads);
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
};

#endif
