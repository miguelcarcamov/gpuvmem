#ifndef MEASUREMENT_OPERATOR_HOST_CUH
#define MEASUREMENT_OPERATOR_HOST_CUH

#include "classes/image.cuh"
#include "image_processing/imageProcessor.cuh"
#include "classes/ckernel.cuh"
#include "ms/metadata.h"

struct varsPerGPU;

/**
 * Non-owning bundle for the forward model’s image-domain inputs (similar in spirit to
 * pyralysis.transformers.MeasurementOperator holding model + dataset context).
 *
 * - I_plane: device brightness (may differ from grid_image->getImage() during line search
 *   or for Stokes plane offsets).
 * - grid_image: grid dimensions and imaging_geometry() (0-based reference column/row, scales).
 * - ip: VirtualImageProcessor (image_count / MFS chain); does not own I_plane.
 */
struct MeasurementGridView {
  float* I_plane{nullptr};
  const Image* grid_image{nullptr};
  VirtualImageProcessor* ip{nullptr};
};

/**
 * Forward model: image domain -> visibility grid on GPU (calculateInu -> baseline PB ->
 * GCF -> FFT2D -> phase_rotate using field phase/reference pixel coordinates).
 *
 * field: MS field metadata; ref_* / phs_* pixel coordinates are filled when the dataset is
 * configured (e.g. MFS::setDevice). Radec→pixel conversion stays there so geometry stays
 * single-sourced.
 */
__host__ void computeImageToVisibilityGridBaseline(
    const MeasurementGridView& model,
    const gpuvmem::ms::FieldMetadata& field,
    varsPerGPU* vars_gpu,
    int gpu_idx,
    float nu,
    float ant1_diameter,
    float ant1_pb_factor,
    float ant1_pb_cutoff,
    int ant1_primary_beam,
    float ant2_diameter,
    float ant2_pb_factor,
    float ant2_pb_cutoff,
    int ant2_primary_beam,
    float fg_scale,
    CKernel* ckernel,
    bool fft_shift);

#endif  // MEASUREMENT_OPERATOR_HOST_CUH
