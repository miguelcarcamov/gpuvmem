#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

__host__ void getNumBlocksAndThreads(int n,
                                     int maxBlocks,
                                     int maxThreads,
                                     int& blocks,
                                     int& threads,
                                     bool reduction);

#endif  // CUDA_UTILS_CUH
