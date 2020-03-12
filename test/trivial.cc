#include "gtest/gtest.h"
#include "motif.h"
#include <fstream>

TEST(MotifTest, trivial) {
  ASSERT_EQ(ooo(), 10000);
}

TEST(MotifTest, simple_markers) {
  std::string source = "1111111111";
  std::vector<std::string> markers = {"1","11","111","1111",
                                      "11111","111111"};
  auto res = match(source, markers);
  ASSERT_EQ(res, "111111");
}

TEST(MotifTest, test_missing_markers) {

  std::string source = "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA";
  std::vector<std::string> markers = {"ACGACTGAGCACT", "GAC", "ACG", "ACGACT", "ACTGTT"};
  auto res = match(source, markers);

  ASSERT_EQ(res, "11110");

  source = "GGCTCAAATTACACGTAAACTTAAGAATATTCG";
  markers = {"TAC", "ACA", "TTACAG", "ATA", "TAT", "ATAT"};
  res = match(source, markers);

  ASSERT_EQ(res, "110111");
}

TEST(MorifTest, test_empty) {
  std::string source = "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA";
  std::vector<std::string> markers = {};
  auto res = match(source, markers);

  ASSERT_EQ(res, "");
}

TEST(MotifTest, test_multiple_small_strings) {
  std::string source = "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA";
  std::vector<std::string> markers = {"11", "123", "125", "ATA"};
  auto res = match(source, markers);

  ASSERT_EQ(res, "0001");
}

TEST(MotifTest, test_cascade) {
  std::string source = "1111111111";
  std::vector<std::string> markers = {"1","12","111","1111"};
  auto res = match(source, markers);

  ASSERT_EQ(res, "101111");
}

