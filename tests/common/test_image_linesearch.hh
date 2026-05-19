#pragma once

#include "classes/image.cuh"
#include "linesearch/linesearch_utils.cuh"

#include <vector>

namespace gpuvmem {
namespace test {

/** MFS-style imageMap so Fixed/Brent line search can evaluate f(x + α d). */
inline std::vector<imageMap> make_linesearch_image_maps(int image_count) {
  std::vector<imageMap> maps(static_cast<size_t>(image_count));
  for (imageMap& m : maps) {
    m.newP = defaultNewP;
    m.evaluateXt = defaultEvaluateXt;
  }
  return maps;
}

inline void wire_image_for_linesearch(Image& image, std::vector<imageMap>& maps) {
  image.setFunctionMapping(maps.data());
}

}  // namespace test
}  // namespace gpuvmem
