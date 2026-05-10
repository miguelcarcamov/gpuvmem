#ifndef FRAMEWORK_CUDA_GRID_CUH
#define FRAMEWORK_CUDA_GRID_CUH

#include <climits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utils/cuda_utils.cuh"
#include "utils/math_utils.hh"

namespace gpuvmem {

namespace detail {
inline long i_div_up(long a, long b) {
  if (b <= 0) return 0;
  return (a + b - 1) / b;
}
}  // namespace detail

/** Primary template: only explicit specializations for Dim in {1,2,3} exist. */
template <int Dim>
class CudaGrid;

/**
 * CUDA launch grid for a 1D index space (e.g. visibility / reduction work).
 * Private members hold the kernel launch configuration; use blocks() / threads()
 * for <<<grid, block>>> and set_* only when adjusting a grid in place.
 */
template <>
class CudaGrid<1> {
 public:
  /** Number of logical index axes this grid covers (compile-time). */
  static constexpr int kDimension = 1;

  __host__ CudaGrid() = default;

  // --- CUDA launch grid (private backing, public accessors) ---
  __host__ const dim3& blocks() const { return blocks_; }
  __host__ const dim3& threads() const { return threads_; }
  __host__ void set_blocks(dim3 b) { blocks_ = b; }
  __host__ void set_threads(dim3 t) { threads_ = t; }

  /** @p n is the total number of logical elements along the 1D axis. */
  static __host__ CudaGrid<1> from_total(long n, int thread_x = 256) {
    CudaGrid<1> g;
    if (thread_x <= 0) thread_x = 256;
    if (n <= 0) {
      g.set_blocks(dim3(0u, 1u, 1u));
      g.set_threads(dim3(1u, 1u, 1u));
      return g;
    }
    g.set_threads(dim3(static_cast<unsigned int>(thread_x), 1u, 1u));
    const long nb = detail::i_div_up(n, thread_x);
    const unsigned int nbu =
        nb > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(nb);
    g.set_blocks(dim3(nbu, 1u, 1u));
    return g;
  }

  /**
   * Heuristic 1D tiling from device limits (warp alignment, max threads/block).
   * For empirical autotuning per kernel, extend later; this matches common
   * visibility-path sizes without requiring CLI -V.
   */
  static __host__ CudaGrid<1> from_auto(long n, const cudaDeviceProp& prop) {
    const int max_t =
        prop.maxThreadsPerBlock > 0 ? prop.maxThreadsPerBlock : 1024;
    const int warp = prop.warpSize > 0 ? prop.warpSize : 32;
    int t = 256;
    if (n <= 0) return from_total(n, warp);
    if (n < 64)
      t = 32;
    else if (n < 256)
      t = 64;
    else if (n < 1024)
      t = 128;
    else if (n < 4096)
      t = 256;
    else if (n < 65536)
      t = 512;
    else
      t = 1024;
    if (t > max_t) t = max_t;
    t = (t / warp) * warp;
    if (t < warp) t = warp;
    return from_total(n, t);
  }

 private:
  dim3 blocks_{};
  dim3 threads_{};
};

/**
 * CUDA launch grid for a 2D index space (e.g. image-sized kernels).
 * blockIdx.x / threadIdx.x cover dim0; blockIdx.y / threadIdx.y cover dim1.
 */
template <>
class CudaGrid<2> {
 public:
  static constexpr int kDimension = 2;

  __host__ CudaGrid() = default;

  __host__ const dim3& blocks() const { return blocks_; }
  __host__ const dim3& threads() const { return threads_; }
  __host__ void set_blocks(dim3 b) { blocks_ = b; }
  __host__ void set_threads(dim3 t) { threads_ = t; }

  static __host__ CudaGrid<2> from_extents(long dim0, long dim1,
                                           dim3 threads_per_block) {
    CudaGrid<2> g;
    int tx = threads_per_block.x > 0 ? static_cast<int>(threads_per_block.x) : 1;
    int ty = threads_per_block.y > 0 ? static_cast<int>(threads_per_block.y) : 1;
    g.set_threads(dim3(static_cast<unsigned int>(tx), static_cast<unsigned int>(ty),
                       1u));
    if (dim0 <= 0 || dim1 <= 0) {
      g.set_blocks(dim3(0u, 0u, 1u));
      return g;
    }
    const long bx = detail::i_div_up(dim0, tx);
    const long by = detail::i_div_up(dim1, ty);
    const unsigned int bxu =
        bx > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(bx);
    const unsigned int byu =
        by > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(by);
    g.set_blocks(dim3(bxu, byu, 1u));
    return g;
  }

  /** Build from precomputed block/thread counts (e.g. legacy heuristics). */
  static __host__ CudaGrid<2> from_blocks_threads(dim3 b, dim3 t) {
    CudaGrid<2> g;
    g.set_blocks(b);
    g.set_threads(dim3(t.x > 0u ? t.x : 1u, t.y > 0u ? t.y : 1u, t.z > 0u ? t.z : 1u));
    return g;
  }

  /**
   * Heuristic 2D tiling matching historical MFS behaviour: getNumBlocksAndThreads
   * per axis with max_threads_axis = sqrt(256) == 16.
   */
  static __host__ CudaGrid<2> from_auto(long dim0, long dim1,
                                        const cudaDeviceProp& prop) {
    (void)prop;
    constexpr int max_threads_axis = 16;
    const int d0 = dim0 > static_cast<long>(INT_MAX) ? INT_MAX : static_cast<int>(dim0);
    const int d1 = dim1 > static_cast<long>(INT_MAX) ? INT_MAX : static_cast<int>(dim1);
    const int max_grid_x = iDivUp(d0, max_threads_axis);
    const int max_grid_y = iDivUp(d1, max_threads_axis);
    int blocks_x = 0, threads_x = 0, blocks_y = 0, threads_y = 0;
    getNumBlocksAndThreads(d0, max_grid_x, max_threads_axis, blocks_x, threads_x,
                           false);
    getNumBlocksAndThreads(d1, max_grid_y, max_threads_axis, blocks_y, threads_y,
                           false);
    CudaGrid<2> g;
    g.set_threads(dim3(static_cast<unsigned int>(threads_x),
                       static_cast<unsigned int>(threads_y), 1u));
    g.set_blocks(dim3(static_cast<unsigned int>(blocks_x),
                      static_cast<unsigned int>(blocks_y), 1u));
    return g;
  }

 private:
  dim3 blocks_{};
  dim3 threads_{};
};

/**
 * CUDA launch grid for a 3D index space.
 * blockIdx.* / threadIdx.* align with dim0, dim1, dim2 respectively.
 */
template <>
class CudaGrid<3> {
 public:
  static constexpr int kDimension = 3;

  __host__ CudaGrid() = default;

  __host__ const dim3& blocks() const { return blocks_; }
  __host__ const dim3& threads() const { return threads_; }
  __host__ void set_blocks(dim3 b) { blocks_ = b; }
  __host__ void set_threads(dim3 t) { threads_ = t; }

  static __host__ CudaGrid<3> from_extents(long d0, long d1, long d2,
                                           dim3 threads_per_block) {
    CudaGrid<3> g;
    int tx = threads_per_block.x > 0 ? static_cast<int>(threads_per_block.x) : 1;
    int ty = threads_per_block.y > 0 ? static_cast<int>(threads_per_block.y) : 1;
    int tz = threads_per_block.z > 0 ? static_cast<int>(threads_per_block.z) : 1;
    g.set_threads(dim3(static_cast<unsigned int>(tx), static_cast<unsigned int>(ty),
                     static_cast<unsigned int>(tz)));
    if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
      g.set_blocks(dim3(0u, 0u, 0u));
      return g;
    }
    const long b0 = detail::i_div_up(d0, tx);
    const long b1 = detail::i_div_up(d1, ty);
    const long b2 = detail::i_div_up(d2, tz);
    const unsigned int b0u =
        b0 > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(b0);
    const unsigned int b1u =
        b1 > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(b1);
    const unsigned int b2u =
        b2 > static_cast<long>(UINT_MAX) ? UINT_MAX : static_cast<unsigned int>(b2);
    g.set_blocks(dim3(b0u, b1u, b2u));
    return g;
  }

  /** Conservative 3D tile: shrink 8×8×4 until product fits maxThreadsPerBlock. */
  static __host__ CudaGrid<3> from_auto(long d0, long d1, long d2,
                                        const cudaDeviceProp& prop) {
    const int max_t =
        prop.maxThreadsPerBlock > 0 ? prop.maxThreadsPerBlock : 256;
    unsigned int tx = 8u, ty = 8u, tz = 4u;
    for (;;) {
      const int p = static_cast<int>(tx * ty * tz);
      if (p <= max_t || (tx <= 4u && ty <= 4u && tz <= 2u)) break;
      if (tx > 4u)
        tx /= 2u;
      else if (ty > 4u)
        ty /= 2u;
      else if (tz > 2u)
        tz /= 2u;
      else if (tx > 1u)
        tx /= 2u;
      else
        break;
    }
    const dim3 th(tx > 0u ? tx : 1u, ty > 0u ? ty : 1u, tz > 0u ? tz : 1u);
    return from_extents(d0, d1, d2, th);
  }

 private:
  dim3 blocks_{};
  dim3 threads_{};
};

}  // namespace gpuvmem

#endif  // FRAMEWORK_CUDA_GRID_CUH
