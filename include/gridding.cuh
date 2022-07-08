#include "framework.cuh"
#include "functions.cuh"

class Gridding : public Filter {
 public:
  Gridding(int threads);
  Gridding();
  void applyCriteria(Visibilities* v);
  void setThreads(int t) { this->threads = t; }
  void configure(void* params);

 private:
  int threads = 1;
};
