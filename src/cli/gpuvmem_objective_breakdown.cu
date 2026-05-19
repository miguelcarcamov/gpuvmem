#include "cli/gpuvmem_objective_breakdown.hh"

#include "classes/fi.cuh"
#include "classes/objectivefunction.cuh"

namespace gpuvmem {
namespace cli {

ObjectiveBreakdown objective_breakdown_from_of(ObjectiveFunction* of, float phi) {
  ObjectiveBreakdown b;
  b.phi = phi;
  if (of == nullptr) return b;

  Fi* chi2 = of->getFiByName("Chi2");
  if (chi2 != nullptr) {
    b.chi2_w = static_cast<float>(static_cast<double>(chi2->getPenalizationFactor()) *
                                  static_cast<double>(chi2->get_fivalue()));
  }
  b.reg_w = phi - b.chi2_w;
  return b;
}

}  // namespace cli
}  // namespace gpuvmem
