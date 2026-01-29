
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <memory>

// Forward declarations
class Image;
class ObjectiveFunction;
class LineSearcher;

class Optimizer {
 public:
  __host__ virtual void allocateMemoryGpu() = 0;
  __host__ virtual void deallocateMemoryGpu() = 0;
  __host__ virtual void optimize() = 0;
  __host__ virtual int getK(){};
  __host__ virtual void setK(int K){};
  //__host__ virtual void configure() = 0;

  __host__ Optimizer() {
    this->ftol = 1E-12;
    this->gtol = 1E-12;
    this->total_iterations = 100;
  };

  __host__ Optimizer(int total_iterations, float ftol) {
    this->ftol = ftol;
    this->gtol = 1E-12;
    this->total_iterations = total_iterations;
  };

  __host__ Optimizer(int total_iterations, float ftol, float gtol) {
    this->ftol = ftol;
    this->gtol = gtol;
    this->total_iterations = total_iterations;
  };

  __host__ float getFtol() { return this->ftol; };

  __host__ float getGtol() { return this->gtol; };

  __host__ int getCurrentIteration() { return this->current_iteration; };

  __host__ void setImage(Image* image) { this->image = image; };
  __host__ void setObjectiveFunction(ObjectiveFunction* of) { this->of = of; };
  void setFlag(int flag) { this->flag = flag; };

  void setFTol(float ftol) { this->ftol = ftol; };

  void setGTol(float gtol) { this->gtol = gtol; };

  void setTotalIterations(int iterations) {
    this->total_iterations = iterations;
  };

  ObjectiveFunction* getObjectiveFunction() { return this->of; };

  /**
   * @brief Set the line search algorithm to use.
   * 
   * @param searcher Line searcher instance (ownership transferred)
   * 
   * Default implementation does nothing. Derived classes (ConjugateGradient, LBFGS)
   * override this to provide actual implementation.
   */
  __host__ virtual void setLineSearcher(std::unique_ptr<LineSearcher> searcher) {}

protected:
  ObjectiveFunction* of;
  Image* image;
  int flag;
  int total_iterations;
  int current_iteration = 0;
  float ftol;
  float gtol;
};

#endif  // OPTIMIZER_CUH
