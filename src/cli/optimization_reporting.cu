#include "cli/optimization_reporting.hh"

#include "cli/gpuvmem_objective_breakdown.hh"
#include "classes/fi.cuh"
#include "classes/objectivefunction.cuh"
#include "classes/optimizer.cuh"

#include <iomanip>

namespace gpuvmem {
namespace cli {

IterationMetrics make_iteration_metrics(Optimizer* optimizer, ObjectiveFunction* of, float phi,
                                      double wall_time_s) {
  IterationMetrics m;
  if (optimizer != nullptr) {
    m.iteration = optimizer->getCurrentIteration();
    m.max_iterations = optimizer->getTotalIterations();
    optimizer->getOptimizationSubRunContext(m.sub_run, m.sub_run_total, m.sub_plane_label);
  }
  m.wall_time_s = wall_time_s;
  const ObjectiveBreakdown b = objective_breakdown_from_of(of, phi);
  m.phi = b.phi;
  m.chi2_w = b.chi2_w;
  m.reg_w = b.reg_w;
  return m;
}

void notify_iteration(IRunObserver* observer, Optimizer* optimizer, ObjectiveFunction* of,
                      float phi, double wall_time_s) {
  if (observer == nullptr) return;
  observer->on_iteration(make_iteration_metrics(optimizer, of, phi, wall_time_s));
}

FinalRunMetrics build_final_run_metrics(Optimizer* optimizer, ObjectiveFunction* of,
                                        float* image_device, const std::string& optimizer_id,
                                        const std::string& optimization_mode, int iteration_budget,
                                        double cpu_time_s, double wall_time_s) {
  FinalRunMetrics m;
  m.optimizer_id = optimizer_id;
  m.optimization_mode = optimization_mode;
  m.iteration_budget = iteration_budget;
  m.cpu_time_s = cpu_time_s;
  m.wall_time_s = wall_time_s;
  if (optimizer != nullptr) {
    m.iteration_last = optimizer->getCurrentIteration();
    m.ftol = static_cast<double>(optimizer->getFtol());
    m.gtol = static_cast<double>(optimizer->getGtol());
  }
  if (of != nullptr && image_device != nullptr) {
    m.phi_total = of->calcFunction(image_device);
    const ObjectiveBreakdown b = objective_breakdown_from_of(of, m.phi_total);
    m.chi2_w = b.chi2_w;
    m.reg_w = b.reg_w;
    for (Fi* fi : of->getFi()) {
      if (fi == nullptr) continue;
      FiMetricLine line;
      line.name = fi->getName();
      line.lambda = static_cast<double>(fi->getPenalizationFactor());
      line.value = static_cast<double>(fi->get_fivalue());
      line.lambda_times_value = line.lambda * line.value;
      m.fi_terms.push_back(line);
    }
  }
  return m;
}

}  // namespace cli
}  // namespace gpuvmem
