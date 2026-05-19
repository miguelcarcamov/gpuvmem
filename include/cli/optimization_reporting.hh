#ifndef GPUVMEM_CLI_OPTIMIZATION_REPORTING_HH
#define GPUVMEM_CLI_OPTIMIZATION_REPORTING_HH

#include "cli/cli_metrics.hh"
#include "cli/run_observer.hh"

class ObjectiveFunction;
class Optimizer;

namespace gpuvmem {
namespace cli {

IterationMetrics make_iteration_metrics(Optimizer* optimizer, ObjectiveFunction* of, float phi,
                                      double wall_time_s);

void notify_iteration(IRunObserver* observer, Optimizer* optimizer, ObjectiveFunction* of,
                    float phi, double wall_time_s);

FinalRunMetrics build_final_run_metrics(Optimizer* optimizer, ObjectiveFunction* of,
                                        float* image_device, const std::string& optimizer_id,
                                        const std::string& optimization_mode, int iteration_budget,
                                        double cpu_time_s, double wall_time_s);

void write_final_metrics(std::ostream& out, const FinalRunMetrics& metrics);

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_OPTIMIZATION_REPORTING_HH
