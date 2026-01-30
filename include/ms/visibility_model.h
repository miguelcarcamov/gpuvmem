#ifndef GPUVMEM_MS_VISIBILITY_MODEL_H
#define GPUVMEM_MS_VISIBILITY_MODEL_H

/**
 * Umbrella header for the MS-aligned visibility model.
 * Include this to get Field, Baseline, TimeSample, SpectralWindow,
 * DataDescription, FieldMetadata, MeasurementSetMetadata, MeasurementSet,
 * DataColumn, MSReader, MSWriter, GriddedVisibilities, VisibilityGPUBuffers.
 *
 * Structure:
 *   MeasurementSet (top-level)
 *   ├── MeasurementSetMetadata (shared: SpectralWindow, DataDescription)
 *   └── Field[] (per field)
 *         ├── FieldMetadata (phase center, reference center, ...)
 *         └── Baseline[] (per baseline)
 *               └── TimeSample[] (per time: uvw, weight, visibilities)
 *
 * I/O: MSReader / MSWriter for DATA, CORRECTED_DATA, MODEL_DATA (and residual).
 * Gridding: GriddedVisibilitySet holds gridded visibilities in the same structure.
 * GPU: VisibilityGPUBuffers for device-side data for synthesis.
 */

#include "ms/metadata.h"
#include "ms/time_sample.h"
#include "ms/baseline.h"
#include "ms/field.h"
#include "ms/measurement_set_metadata.h"
#include "ms/measurement_set.h"
#include "ms/data_column.h"
#include "ms/ms_reader.h"
#include "ms/ms_writer.h"
#include "ms/gridded_visibilities.h"
#include "ms/gpu_buffers.h"
#include "ms/primary_beam.h"

#endif  // GPUVMEM_MS_VISIBILITY_MODEL_H
