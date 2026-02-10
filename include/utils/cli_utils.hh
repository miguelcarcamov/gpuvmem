#ifndef CLI_UTILS_H
#define CLI_UTILS_H

// Include framework.cuh for Vars definition (it's now safe for pure C++ code)
#include "framework.cuh"

void print_help();
Vars getOptions(int argc, char** argv);

#endif  // CLI_UTILS_H
