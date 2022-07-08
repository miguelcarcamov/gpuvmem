#ifndef FACTORY_CUH
#define FACTORY_CUH

#include <boost/type_index.hpp>
#include <exception>
#include <iostream>

template <class T>
class Singleton {
 public:
  static T& Instance() {
    static T instance;
    return instance;
  }

 private:
  Singleton(){};
  Singleton(T const&) = delete;
  void operator=(T const&) = delete;
};

template <class IdentifierType, class ProductType>
class DefaultFactoryError {
 public:
  class Exception : public std::exception {
   public:
    Exception(const IdentifierType& unknownId) : unknownId_(unknownId) {}
    virtual const char* what() const noexcept {
      return "Unknown object type passed to Factory";
    }
    const IdentifierType& GetId() { return unknownId_; };

   private:
    IdentifierType unknownId_;
  };

 protected:
  static ProductType* OnUnknownType(const IdentifierType& id) {
    throw Exception(id);
  }
};

template <class AbstractProduct, class IdentifierType,
          class ProductCreator = AbstractProduct* (*)(),
          template <typename, class> class FactoryErrorPolicy =
              DefaultFactoryError>
class Factory : public FactoryErrorPolicy<IdentifierType, AbstractProduct> {
 public:
  bool Register(const IdentifierType& id, ProductCreator creator) {
    return associations_
        .insert(
            typename Factory<AbstractProduct, IdentifierType, ProductCreator,
                             FactoryErrorPolicy>::AssocMap::value_type(id,
                                                                       creator))
        .second;
  }
  bool Unregister(const IdentifierType& id) {
    return associations_.erase(id) == 1;
  }
  AbstractProduct* CreateObject(const IdentifierType& id) {
    typename AssocMap::const_iterator i = associations_.find(id);
    if (i != associations_.end()) {
      return (i->second)();
    }
    return this->OnUnknownType(id);
  }

 private:
  typedef std::map<IdentifierType, ProductCreator> AssocMap;
  AssocMap associations_;
};

template <class T, class V>
T* createObject(V value) {
  try {
    return Singleton<Factory<T, V>>::Instance().CreateObject(value);
  } catch (std::exception& e) {
    std::cerr << e.what() << " of class "
              << boost::typeindex::type_id<T>().pretty_name()
              << " and missing id: " << value << std::endl;
    exit(-1);
  }
};

template <class T, class V, class Creator = T* (*)()>
bool registerCreationFunction(V value, Creator function) {
  return Singleton<Factory<T, V>>::Instance().Register(value, function);
}
#endif FACTORY_CUH
