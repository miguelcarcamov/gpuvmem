#ifndef GPUVMEM_CLI_GPUVMEM_CLI_CONFIG_HH
#define GPUVMEM_CLI_GPUVMEM_CLI_CONFIG_HH

#include "framework/vars.hh"

#include <iosfwd>
#include <string>

/** Boolean run flags (legacy code still reads file-scope globals synced from this). */
struct GpuvmemCliRuntimeFlags {
  bool verbose = false;
  bool quiet = false;
  bool debug = false;
  bool nopositivity = false;
  bool apply_noise = false;
  bool print_images = false;
  bool print_errors = false;
  bool save_model_input = false;
  bool radius_mask = false;
  bool modify_weights = false;
  enum ProgressModeKind { ProgressAuto = 0, ProgressBar, ProgressPlain, ProgressOff };
  std::string progress_mode_user = "auto";
  ProgressModeKind progress_mode = ProgressAuto;
  int log_interval = 1;
  /** If set, print GPL warranty text and exit (no CUDA / no imaging run). */
  bool print_warranty = false;
  /** If set, print GPL copying conditions and exit (no CUDA / no imaging run). */
  bool print_copyright = false;
};

/** Full CLI parse result: scalar/string options plus runtime booleans. */
struct GpuvmemCliConfig {
  Vars vars{};
  GpuvmemCliRuntimeFlags runtime{};
  /** argv[0] from parse (for log messages); empty if unset. */
  std::string argv0;
};

bool parse_gpuvmem_cli(int argc, char** argv, GpuvmemCliConfig& out,
                       std::ostream& err);

/**
 * Points at the runtime flags object owned by the active synthesizer config
 * (e.g. MFS::cli_config_.runtime). Set from MFS::syncLegacyGlobalsFromCli_;
 * nullptr before configure.
 */
const GpuvmemCliRuntimeFlags* gpuvmem_cli_runtime_ptr();
void gpuvmem_cli_runtime_bind(const GpuvmemCliRuntimeFlags* runtime);

inline bool gpuvmem_cli_verbose() {
  const GpuvmemCliRuntimeFlags* p = gpuvmem_cli_runtime_ptr();
  return p != nullptr && (p->verbose || p->debug);
}

inline bool gpuvmem_cli_debug() {
  const GpuvmemCliRuntimeFlags* p = gpuvmem_cli_runtime_ptr();
  return p != nullptr && p->debug;
}

inline bool gpuvmem_cli_quiet() {
  const GpuvmemCliRuntimeFlags* p = gpuvmem_cli_runtime_ptr();
  return p != nullptr && p->quiet;
}

inline bool gpuvmem_cli_nopositivity() {
  const GpuvmemCliRuntimeFlags* p = gpuvmem_cli_runtime_ptr();
  return p != nullptr && p->nopositivity;
}

#endif  // GPUVMEM_CLI_GPUVMEM_CLI_CONFIG_HH
