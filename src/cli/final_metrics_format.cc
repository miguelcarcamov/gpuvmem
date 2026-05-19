#include "cli/optimization_reporting.hh"

#include <iomanip>
#include <ostream>

namespace gpuvmem {
namespace cli {

void write_final_metrics(std::ostream& sink, const FinalRunMetrics& metrics) {
  sink << "\n=== gpuvmem final metrics ===\n";
  sink << "# phi_total = sum(lambda_i * value_i). phi_check = chi2_w + reg_w.\n";
  sink << std::defaultfloat << std::setprecision(9) << "Objective_phi_total: "
       << static_cast<double>(metrics.phi_total) << '\n';
  sink << std::scientific << std::setprecision(6)
       << "phi_check_chi2_w: " << static_cast<double>(metrics.chi2_w) << '\n';
  sink << "phi_check_reg_w: " << static_cast<double>(metrics.reg_w) << '\n';
  sink << std::defaultfloat;
  sink << "Optimizer_id: " << metrics.optimizer_id << '\n';
  sink << "Optimization_mode: " << metrics.optimization_mode << '\n';
  sink << std::setprecision(9) << "Ftol: " << metrics.ftol << "  Gtol: " << metrics.gtol << '\n';
  sink << "Iteration_budget: " << metrics.iteration_budget << '\n';
  sink << "Iteration_last: " << metrics.iteration_last << '\n';
  sink << "Fi_term_count: " << metrics.fi_terms.size() << '\n';
  for (size_t i = 0; i < metrics.fi_terms.size(); ++i) {
    const FiMetricLine& line = metrics.fi_terms[i];
    sink << std::setprecision(9) << "Fi[" << i << "] name=" << line.name << " lambda=" << line.lambda
         << " value=" << line.value << " lambda_times_value=" << line.lambda_times_value << '\n';
  }
  sink << std::fixed << std::setprecision(6) << "Cpu_time_s: " << metrics.cpu_time_s << '\n'
       << "Wall_time_s: " << metrics.wall_time_s << '\n';
  sink << "=== end summary ===\n\n";
}

}  // namespace cli
}  // namespace gpuvmem
