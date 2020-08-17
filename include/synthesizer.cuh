#ifndef ALPHAMFS_CUH
#define ALPHAMFS_CUH

#include "framework.cuh"
#include "functions.cuh"
#include "frprmn.cuh"
#include <time.h>
#include "directioncosines.cuh"

class MFS : public Synthesizer
{
public:
void writeImages();
void clearRun();
void writeResiduals();
void run();
void setOutPut(char * FileName){
};
void setDevice();
void unSetDevice();
std::vector<std::string> countAndSeparateStrings(char *input);
void configure(int argc, char **argv);
void applyFilter(Filter *filter){
        filter->applyCriteria(this->visibilities);
};
};

#endif
