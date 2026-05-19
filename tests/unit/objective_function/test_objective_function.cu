#include <gtest/gtest.h>

#include "classes/fi.cuh"
#include "classes/objectivefunction.cuh"

namespace {

class DummyFi : public Fi {
 public:
  explicit DummyFi(float lambda) { setPenalizationFactor(lambda); }
  float calcFi(float*) override { return 0.f; }
  void calcGi(float*, float*) override {}
  void restartDGi() override {}
  void addToDphi(float*) override {}
  float calculateSecondDerivate() override { return 0.f; }
};

}  // namespace

TEST(ObjectiveFunction, AddFiSkipsZeroLambda) {
  ObjectiveFunction of;
  DummyFi active(1.f);
  DummyFi inactive(0.f);
  active.attachToObjectiveFunction(&of);
  inactive.attachToObjectiveFunction(&of);
  of.setGridDimensions(4, 4, 1);
  of.addFi(&active);
  of.addFi(&inactive);
  EXPECT_EQ(of.getFi().size(), 1u);
}

TEST(ObjectiveFunction, GetFiByNameFindsRegisteredTerm) {
  ObjectiveFunction of;
  DummyFi term(1.f);
  term.setName("dummy_term");
  term.attachToObjectiveFunction(&of);
  of.setGridDimensions(4, 4, 1);
  of.addFi(&term);
  EXPECT_NE(of.getFiByName("dummy_term"), nullptr);
  EXPECT_EQ(of.getFiByName("missing"), nullptr);
}

TEST(ObjectiveFunction, RegularizationWeightsStored) {
  ObjectiveFunction of;
  float w[] = {1.f, 0.1f, 0.01f};
  of.setRegularizationWeights(w, 3);
  EXPECT_EQ(of.getRegularizationWeightCount(), 3);
  ASSERT_NE(of.getRegularizationWeights(), nullptr);
  EXPECT_FLOAT_EQ(of.getRegularizationWeights()[1], 0.1f);
}
