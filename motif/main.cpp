#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <thread>
#include <mutex>
#include "motif.h"
#include "file_readers.h"


#define REQUIRED_ARGS_AMOUNT        4
#define GENOMES_FILE_ARG            1
#define MARKERS_FILE_ARG            2
#define OUTPUT_FILE_ARG             3
#define THREAD_NUM_ARG              4


inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_us(const D& d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}


int main(int argc, char** argv) {
    if (argc != REQUIRED_ARGS_AMOUNT) {
        std::cerr << "Wrong arguments amount (expected 4). Usage: \n"
                     "<program> <genomes-filename> <markers-filename>"
                     "<output-filename>\n" << std::endl;
        return 1;
    }

    const uint8_t threads_num = 4;

    auto start = get_current_time_fenced();


    std::string f_genomes_path(argv[GENOMES_FILE_ARG]);
    // read genomes paths
    std::ifstream f_genomes(f_genomes_path);
    if (!f_genomes.is_open()) {
        std::cerr << "Cannot read file: " << argv[GENOMES_FILE_ARG] << std::endl;
        return 1;
    }
    std::vector<std::deque<std::string>> genomes_paths(threads_num);
    read_genome_paths(f_genomes, genomes_paths);

    f_genomes.close();


    // read markers

    std::ifstream f_markers(argv[MARKERS_FILE_ARG]);
    if (!f_markers.good()) {
        std::cerr << "Cannot read file: " << argv[MARKERS_FILE_ARG] << std::endl;
        return 1;
    }
    std::deque<std::string> markers;
    const auto markersData = read_markers(f_markers, markers);

    f_markers.close();


    auto end = get_current_time_fenced();

    std::cout << "Files read: " << to_us(end - start) << std::endl;


    start = get_current_time_fenced();

    auto log_4_markers_size = std::log(markers.size()) / std::log(4);

    uint32_t max_states_num = (
          (std::pow(4, std::ceil(log_4_markers_size)) - 1) / (4 - 1)
    ) + markersData.sum_of_all_chars - std::floor(log_4_markers_size) * markers.size() + 3;


    std::vector<std::array<uint32_t, MATRIX_WIDTH>> automaton(
        max_states_num,
        {0, 0, 0, 0}
    );
    std::vector<std::vector<uint32_t>> output_links(max_states_num);

    create_automaton(markers, automaton, output_links);

    end = get_current_time_fenced();

    std::cout << "Automaton created: " << to_us(end - start) << std::endl;


    // find matches
    start = get_current_time_fenced();

    std::ofstream result(argv[OUTPUT_FILE_ARG]);

    if (!result.good()) {
        std::cerr << "Cannot open file: " << argv[OUTPUT_FILE_ARG] << std::endl;
        return 1;
    }

    std::vector<std::thread> threads_vector;

    std::mutex file_write_mutex;

    for (uint8_t i = 0; i < threads_num; ++i) {
        threads_vector.emplace_back(
            match_genomes,
            std::cref(genomes_paths[i]),
            f_genomes_path,
            markers.size(),
            std::cref(automaton),
            std::cref(output_links),
            std::ref(file_write_mutex),
            std::ref(result)
        );
    }



    for (auto& thread_: threads_vector) {
        thread_.join();
    }

    result.close();
    end = get_current_time_fenced();

    std::cout << "Genomes analized: " << to_us(end - start) << std::endl;

    return 0;
}
