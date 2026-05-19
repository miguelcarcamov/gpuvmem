#ifndef MAIN_CUH
#define MAIN_CUH

#include "cli/gpuvmem_cli_config.hh"

/** Print usage and exit (used on missing/invalid options). */
/** Print CLI summary and exit. Pass argv[0] when available for a correct Usage line. */
void print_help(const char* program_name = nullptr);

/** Exit with error (used on runtime failures). */
void goToError();

#endif  // MAIN_CUH
