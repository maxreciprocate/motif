#include <cstdint>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <fstream>

#ifndef _FILE_READERS_H
#define _FILE_READERS_H

struct markersData {
    uint64_t sum_of_all_chars;
    uint32_t longest_marker_len;
};

markersData read_markers(std::ifstream &in, std::deque<std::string> &container);

void read_genome_paths(std::ifstream &in, std::vector<std::deque<std::string>> &container);

void read_genome_file(const std::string &file_name, std::string &container);

#endif //_FILE_READERS_H
