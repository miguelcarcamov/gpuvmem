#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <memory>

#include "optimization/projection.hh"

class Image;
class ObjectiveFunction;
class LineSearcher;

class Optimizer {
 public:
  __host__ virtual ~Optimizer();

  __host__ virtual void allocateMemoryGpu() = 0;
  __host__ virtual void deallocateMemoryGpu() = 0;
  __host__ virtual void optimize() = 0;
  __host__ virtual int getK();
  __host__ virtual void setK(int K);
  //__host__ virtual void configure() = 0;

  __host__ Optimizer();
  __host__ Optimizer(int total_iterations, float ftol);
  __host__ Optimizer(int total_iterations, float ftol, float gtol);

  __host__ float getFtol();
  __host__ float getGtol();
  __host__ int getCurrentIteration();

  __host__ void setImage(Image* image);
  __host__ void setObjectiveFunction(ObjectiveFunction* of);
  void setFlag(int flag);

  void setFTol(float ftol);
  void setGTol(float gtol);
  void setTotalIterations(int iterations);

  ObjectiveFunction* getObjectiveFunction();

  __host__ virtual void setLineSearcher(std::unique_ptr<LineSearcher> searcher);

  __host__ virtual void setProjection(std::unique_ptr<Projection> projection);

 protected:
  ObjectiveFunction* of;
  Image* image;
  int flag;
  int total_iterations;
  int current_iteration;
  float ftol;
  float gtol;
};

#endif  // OPTIMIZER_CUH
