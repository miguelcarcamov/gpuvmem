#include <gtest/gtest.h>

#include "framework/cuda_grid.cuh"

using gpuvmem::CudaGrid;

TEST(CudaGrid, FromExtents2dComputesBlocks) {
  const CudaGrid<2> grid = CudaGrid<2>::from_extents(64, 32, dim3(16, 16, 1));
  EXPECT_EQ(grid.threads().x, 16u);
  EXPECT_EQ(grid.threads().y, 16u);
  EXPECT_EQ(grid.blocks().x, 4u);
  EXPECT_EQ(grid.blocks().y, 2u);
}

TEST(CudaGrid, ZeroExtentsYieldZeroBlocks) {
  const CudaGrid<2> grid = CudaGrid<2>::from_extents(0, 8, dim3(8, 8, 1));
  EXPECT_EQ(grid.blocks().x, 0u);
}

TEST(CudaGrid1d, FromTotal) {
  const CudaGrid<1> grid = CudaGrid<1>::from_total(1000, 256);
  EXPECT_GE(grid.blocks().x, 1u);
  EXPECT_EQ(grid.threads().x, 256u);
}
