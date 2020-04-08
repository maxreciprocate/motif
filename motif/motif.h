#include <string_view>
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <queue>
#include <bitset>
#include <mutex>

#define MATRIX_WIDTH            4
#define MATRIX_WIDTH_LEFT_SHIFT 2u
#define AUTOMATON_WIDTH         4  // since last column stores links to markers

#define START_STATE             1u
#define UNDEFINED_STATE         0

#define ALPHABET_SIZE           26u
#define BITS_PER_BYTE           8u

#ifndef MOTIF
#define MOTIF

#include "src/readers/archive_reader.h"
void match(
    const std::string_view &source,
    const std::vector<uint32_t>& automaton,
    const std::vector<std::vector<uint32_t>>& output_links,
    std::string &result
);

void create_automaton(
    const std::deque<std::string> &markers,
    std::vector<uint32_t> &matrix,
    std::vector<std::vector<uint32_t>> &output_links
);

//
//void match_genome(
//    const file_entry &genome,
//    uint64_t markers_size,
//    const std::vector<uint32_t> &automaton,
//    const std::vector<std::vector<uint32_t>> &output_links,
//    std::mutex &file_write_mutex,
//    std::string& result,
//    std::ofstream &output_file
//);


#endif