#include "projection/projection_from_cli.hh"

std::unique_ptr<Projection> makeLineSearchProjectionFromCli(
    const GpuvmemCliConfig& cfg, const std::vector<float>& xt_reference_per_image,
    const std::vector<float>& minimal_pixel_values_per_image) {
  if (cfg.runtime.nopositivity) {
    return std::make_unique<NoProjection>();
  }
  return std::make_unique<PositivityProjection>(cfg.vars.eta, xt_reference_per_image,
                                                  minimal_pixel_values_per_image);
}
