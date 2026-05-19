#ifndef GPUVMEM_CLI_GPUVMEM_OBJECTIVE_BREAKDOWN_HH
#define GPUVMEM_CLI_GPUVMEM_OBJECTIVE_BREAKDOWN_HH

class ObjectiveFunction;

namespace gpuvmem {
namespace cli {

struct ObjectiveBreakdown {
  float phi = 0.f;
  float chi2_w = 0.f;
  float reg_w = 0.f;
};

/** phi is the accepted objective value; chi2_w/reg_w come from Fi caches after line search. */
ObjectiveBreakdown objective_breakdown_from_of(ObjectiveFunction* of, float phi);

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_GPUVMEM_OBJECTIVE_BREAKDOWN_HH
