#include <sstream>
#include "file_readers.h"

markersData read_markers(std::ifstream &in, std::deque <std::string> &container) {
    uint64_t sum_of_all_chars = 0;
    uint32_t longest_marker_len = 0;
    std::string line;

    while (std::getline(in, line)) {
        container.emplace_back(
            line.begin() + line.find(',') + 1, line.end()
        );

        longest_marker_len = std::max(longest_marker_len, static_cast<uint32_t>(line.size()));
        sum_of_all_chars += line.size();
    }

    return {
        .sum_of_all_chars = sum_of_all_chars,
        .longest_marker_len = longest_marker_len
    };
}

void read_genome_paths(std::ifstream &in, std::vector <std::deque<std::string>> &container) {
    std::string line;

    uint8_t index = 0;

    while (std::getline(in, line)) {
        container[index++].emplace_back(
            line.begin(), line.end()
        );

        if (index == container.size()) {
            index = 0;
        }
    }
}


auto ignore_file(std::ifstream& in, const std::string_view& file_name) {
  if (!in.good()) {
    std::ostringstream msg;
    msg << "Cannot read file: " << file_name;
    throw std::runtime_error(msg.str());
  }

  auto const start_pos = in.tellg();

  if (std::streamsize(-1) == start_pos) {
    throw std::runtime_error("File stream is empty");
  }

  if (!in.ignore(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("Can't ignore() given file");
  }

  auto const char_count = in.gcount();

  if (!in.seekg(start_pos)) {
    throw std::runtime_error("Can't go back to the begging of the file");
  }

  return char_count;
}

void read_genome_file(const std::string& file_name, std::string &container) {
  std::ifstream in(file_name, std::ifstream::binary);

//  auto const char_count = ignore_file(in, file_name);
  // get length of file:
  in.seekg (0, in.end);
  auto const char_count = in.tellg();
  in.seekg (0, in.beg);

  container.resize(char_count);

  if (!in.read(container.data(), char_count)) {
    throw std::runtime_error("Can't read file");
  }

  in.close();
}
