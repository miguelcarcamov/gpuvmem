#pragma once

#include "imaging_geometry.hh"

namespace gpuvmem {
namespace test {

/** Push ImagingGeometryParam into legacy TU globals (deltau, deltav, nu_0, …). */
void apply_legacy_imaging_globals(const ImagingGeometryParam& geom);

}  // namespace test
}  // namespace gpuvmem
