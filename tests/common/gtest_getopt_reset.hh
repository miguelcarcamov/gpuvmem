#pragma once

#include <unistd.h>

#ifdef __APPLE__
#include <cstdlib>
extern int optreset;
#endif

/** Reset libc getopt state between GTest cases (optind persists globally). */
inline void gpuvmem_gtest_reset_getopt() {
  optind = 1;
#ifdef __APPLE__
  optreset = 1;
#endif
  opterr = 1;
}
