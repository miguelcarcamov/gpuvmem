#ifndef MAIN_CUH
#define MAIN_CUH

#include "cli/gpuvmem_cli_config.hh"

/** Print usage and exit (used on missing/invalid options). */
void print_help();

/** Exit with error (used on runtime failures). */
void goToError();

#endif  // MAIN_CUH
