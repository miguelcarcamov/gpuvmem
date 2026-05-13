#ifndef GPUVMEM_PROJECTION_FROM_CLI_HH
#define GPUVMEM_PROJECTION_FROM_CLI_HH

#include "cli/gpuvmem_cli_config.hh"
#include "projection/projection.hh"

#include <memory>
#include <vector>

/**
 * Builds the line-search `Projection` from parsed CLI only: `NoProjection` when
 * `-nopositivity` / `runtime.nopositivity`, otherwise `PositivityProjection` with
 * `vars.eta`, reference levels, and per-image minimal floors.
 */
std::unique_ptr<Projection> makeLineSearchProjectionFromCli(
    const GpuvmemCliConfig& cfg, const std::vector<float>& xt_reference_per_image,
    const std::vector<float>& minimal_pixel_values_per_image);

#endif
