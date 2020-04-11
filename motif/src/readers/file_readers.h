#include <cstdint>
#include <string>
#include <string_view>
#include <deque>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <filesystem>
#include <memory>

#include "archive_reader.h"
#ifndef _FILE_READERS_H
#define _FILE_READERS_H

struct markersData {
    uint64_t sum_of_all_chars;
    uint32_t longest_marker_len;
};

markersData read_markers(std::ifstream &in, std::deque<std::string> &container);

void read_genome_paths(const std::string &file_name, std::vector<std::string> &container);

void read_file(const std::string &file_name, std::string &container);
void read_file(const std::string& file_name, std::string_view &container);
void read_file(const std::string& file_name, file_entry &container);

#endif //_FILE_READERS_H
