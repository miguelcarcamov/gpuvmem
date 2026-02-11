#ifndef MAIN_CUH
#define MAIN_CUH

#include "framework.cuh"  // Vars

/** Parse command-line options (used by MFS synthesizer). */
Vars getOptions(int argc, char** argv);

/** Print usage and exit (used on missing/invalid options). */
void print_help();

/** Exit with error (used on runtime failures). */
void goToError();

#endif  // MAIN_CUH
