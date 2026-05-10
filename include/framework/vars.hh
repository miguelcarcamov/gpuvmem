#ifndef GPUVMEM_FRAMEWORK_VARS_HH
#define GPUVMEM_FRAMEWORK_VARS_HH

#include <string>

/** CLI / run parameters (strings, ints, floats) parsed from argv or filled by tests. */
struct Vars {
  std::string input;
  std::string output;
  std::string inputdat;
  std::string modin;
  std::string ofile;
  std::string path;
  std::string output_image;
  std::string gpus;
  std::string initial_values;
  std::string stokes;  // Stokes to image, e.g. "I" or "I,Q,U,V". Empty = use initial_values count (MFS).
  std::string penalization_factors;
  std::string user_mask;
  int blockSizeX;
  int blockSizeY;
  int blockSizeV;
  int it_max;
  int gridding;
  float noise;
  float noise_cut;
  float randoms;
  float eta;
  float nu_0;
  float robust_param;
  float threshold;
  float alpha_n_sigma;  // N sigma for alpha mask: alpha=0 where I_nu_0 <
                        // alpha_n_sigma*noise (e.g. 3 or 5)
  bool normalize;
  std::string optimization_mode;  // "joint" | "block" | "one" | "alpha_static"
  // joint = I_nu_0 + alpha together; block = alternate I_nu_0, alpha, ...;
  // one = single image / one run; alpha_static = two images, alpha fixed, I_nu_0 only
};

#endif  // GPUVMEM_FRAMEWORK_VARS_HH
