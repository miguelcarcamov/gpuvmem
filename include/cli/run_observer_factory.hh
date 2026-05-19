#ifndef GPUVMEM_CLI_RUN_OBSERVER_FACTORY_HH
#define GPUVMEM_CLI_RUN_OBSERVER_FACTORY_HH

#include "cli/gpuvmem_cli_config.hh"
#include "cli/run_observer.hh"

#include <memory>

namespace gpuvmem {
namespace cli {

std::unique_ptr<IRunObserver> create_run_observer(const GpuvmemCliRuntimeFlags& runtime);

}  // namespace cli
}  // namespace gpuvmem

#endif  // GPUVMEM_CLI_RUN_OBSERVER_FACTORY_HH
