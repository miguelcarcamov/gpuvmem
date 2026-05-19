/* Compilation unit for the MS-aligned visibility model.
 * Includes the header so the model is compiled; reader/factory can be added here later.
 */
#include "ms/visibility_model.h"

namespace gpuvmem {
namespace ms {

// Placeholder: future MS reader will populate MeasurementSet from MAIN table.
// Kept so this .cu compiles and links the visibility model.
void visibility_model_placeholder() {}

}  // namespace ms
}  // namespace gpuvmem
