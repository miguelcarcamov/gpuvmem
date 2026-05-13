#ifndef GPUVMEM_FRAMEWORK_VARS_HH
#define GPUVMEM_FRAMEWORK_VARS_HH

#include <string>

/** CLI / run parameters (strings, ints, floats) parsed from argv or filled by tests. */
struct Vars {
  std::string input;
  std::string output;
  std::string modin;
  std::string ofile;
  std::string path;
  std::string output_image;
  std::string gpus;
  std::string initial_values;
  /** Optional FITS (2D float primary HDU): load plane-0 pixels after -z fill; same M,N as model grid. */
  std::string initial_model_fits;
  std::string stokes;  // Stokes to image, e.g. "I" or "I,Q,U,V". Empty = use initial_values count (MFS).
  /** Comma-separated weights for Fi terms (χ² + regularizers); CLI: -Z / --regularization_factors. */
  std::string regularization_weights;
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

  /** natural|uniform|radial|briggs|robust (robust ≡ Briggs; use -R) */
  std::string weighting_scheme;
  /** Registered Optimizer name, e.g. LBFGS, CG-PolakRibiere */
  std::string optimizer_name;
  /** LineSearcher name, or empty for optimizer default */
  std::string linesearch_name;
  /** StepSizeSeeder name, or empty; if set without linesearch, Brent is used */
  std::string seeder_name;
  /** L-BFGS memory pairs (other optimizers ignore) */
  int lbfgs_corrections;
  /** Comma-separated image size "M,N" (naxis2,naxis1) when no model FITS */
  std::string imsize;
  /** Pixel size in arcsec (square) for synthetic grid without model FITS */
  float cellsize_arcsec;
  /** Phase center "ra_deg,dec_deg" for synthetic grid (ICRS degrees) */
  std::string phase_center_deg;
};

#endif  // GPUVMEM_FRAMEWORK_VARS_HH
