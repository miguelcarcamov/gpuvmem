#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <memory>

#include "projection/projection.hh"

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
  /**
   * Objective stopping: scale-invariant on |f|,
   *   |f_new − f_prev| ≤ ftol · (1 + max(|f_new|, |f_prev|)).
   * Also stops on exact float equality of f (line search made no representable progress).
   * Returns false if either value is non-finite.
   */
  __host__ bool objectiveSequenceWithinTolerance(float f_new, float f_prev) const;

  ObjectiveFunction* of;
  Image* image;
  int flag;
  int total_iterations;
  int current_iteration;
  float ftol;
  float gtol;
};

#endif  // OPTIMIZER_CUH
