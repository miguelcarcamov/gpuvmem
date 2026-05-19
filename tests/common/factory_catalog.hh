#pragma once

#include <string>
#include <vector>

namespace gpuvmem {
namespace test {

/** Fi terms that factory-create without pulling Chi2/MS/beam (unit factory test). */
inline const std::vector<std::string>& fi_factory_light_ids() {
  static const std::vector<std::string> kIds = {
      "Entropy",       "L1-Norm",       "TotalSquaredVariation",
      "Laplacian",     "Quadratic",     "L2ConstantPrior",
      "IsotropicTotalVariation", "AnisotropicTotalVariation"};
  return kIds;
}

inline const std::vector<std::string>& fi_ids() {
  static const std::vector<std::string> kIds = {
      "Chi2",           "Entropy",       "L1-Norm",       "TotalSquaredVariation",
      "Laplacian",      "Quadratic",     "L2ConstantPrior",
      "IsotropicTotalVariation", "AnisotropicTotalVariation",
      "GL1Norm",        "GEntropy"};
  return kIds;
}

inline const std::vector<std::string>& production_fi_ids() {
  static const std::vector<std::string> kIds = {"Chi2", "Entropy", "L1-Norm",
                                                "TotalSquaredVariation"};
  return kIds;
}

inline const std::vector<std::string>& optimizer_ids() {
  static const std::vector<std::string> kIds = {
      "LBFGS",           "CG-PolakRibiere",  "CG-HagerZhang", "CG-DaiYuan",
      "CG-FletcherReeves", "CG-HestenesStiefel", "CG-LiuStorey", "CG-RMIL"};
  return kIds;
}

inline const std::vector<std::string>& linesearch_ids() {
  static const std::vector<std::string> kIds = {
      "Brent", "GoldenSectionSearch", "FibonacciSearch", "GLLArmijo",
      "BacktrackingArmijo", "FistaBacktracking", "Fixed"};
  return kIds;
}

inline const std::vector<std::string>& seeder_ids() {
  static const std::vector<std::string> kIds = {
      "BBMin1Seeder", "BBMin2Seeder", "BBAlternatingSeeder",
      "QuadraticInterpolationSeeder", "CubicInterpolationSeeder"};
  return kIds;
}

inline const std::vector<std::string>& weighting_ids() {
  static const std::vector<std::string> kIds = {"Uniform", "Natural", "Radial",
                                                "Briggs"};
  return kIds;
}

inline const std::vector<std::string>& ckernel_ids() {
  static const std::vector<std::string> kIds = {"PillBox2D", "Gaussian2D", "Sinc2D",
                                                "GaussianSinc2D", "PSWF"};
  return kIds;
}

inline const std::vector<std::string>& io_ids() {
  static const std::vector<std::string> kIds = {"IoMS", "IoFITS"};
  return kIds;
}

inline const std::vector<std::string>& synthesizer_ids() {
  static const std::vector<std::string> kIds = {"MFS"};
  return kIds;
}

}  // namespace test
}  // namespace gpuvmem
