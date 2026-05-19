#include "cli/run_observer_factory.hh"

#include "cli/console_run_observer.hh"

namespace gpuvmem {
namespace cli {

std::unique_ptr<IRunObserver> create_run_observer(const GpuvmemCliRuntimeFlags& runtime) {
  (void)runtime;
  return std::make_unique<ConsoleRunObserver>(runtime);
}

}  // namespace cli
}  // namespace gpuvmem
