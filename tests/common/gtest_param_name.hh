#pragma once

#include <string>

namespace gpuvmem {
namespace test {

/** GTest INSTANTIATE_TEST_SUITE_P names: letters, digits, underscore only. */
inline std::string gtest_sanitize_param_name(std::string name) {
  for (char& c : name) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      continue;
    }
    c = '_';
  }
  if (name.empty() || (name[0] >= '0' && name[0] <= '9')) {
    name = "p_" + name;
  }
  return name;
}

}  // namespace test
}  // namespace gpuvmem
