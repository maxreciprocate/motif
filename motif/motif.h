#include <string_view>
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <queue>
#include <bitset>
#include <mutex>

#define MATRIX_WIDTH        4
#define AUTOMATON_WIDTH     4  // since last column stores links to markers
#define FAIL_COLUMN         4
#define OUTPUT_COLUMN       5

#define START_STATE         1
#define UNDEFINED_STATE     0

#define ALPHABET_SIZE       26u
#define BITS_PER_BYTE       8u

#ifndef MOTIF
#define MOTIF

void match(
    const std::string &source,
    const std::vector<std::array<uint32_t, MATRIX_WIDTH>>& automaton,
    const std::vector<std::vector<uint32_t>>& output_links,
    std::string& result
);

void create_automaton(
    const std::deque<std::string> &markers,
    std::vector<std::array<uint32_t, MATRIX_WIDTH>> &matrix,
    std::vector<std::vector<uint32_t>> &output_links
);


void match_genomes (
    const std::deque<std::string> &genome_paths,
    const std::string& f_genomes_path,
    uint64_t markers_size,
    const std::vector<std::array<uint32_t, MATRIX_WIDTH>>& automaton,
    const std::vector<std::vector<uint32_t>>& output_links,
    std::mutex& file_write_mutex,
    std::ofstream& output_file
);

#endif