#include "gtest/gtest.h"

#define TESTING

class MotifTest : public testing::Test {
  void SetUp() {}

  void TearDown() {}
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
