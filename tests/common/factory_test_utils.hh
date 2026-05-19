#pragma once

#include "factory.cuh"

#include <memory>
#include <string>

namespace gpuvmem {
namespace test {

/** Factory lookup without calling exit() (production createObject aborts on miss). */
template <class Product, class Id>
Product* try_create(const Id& id) {
  try {
    return Singleton<Factory<Product, Id>>::Instance().CreateObject(id);
  } catch (const std::exception&) {
    return nullptr;
  }
}

template <class Product, class Id>
std::unique_ptr<Product> make_unique_product(const Id& id) {
  Product* raw = try_create<Product, Id>(id);
  return std::unique_ptr<Product>(raw);
}

}  // namespace test
}  // namespace gpuvmem
