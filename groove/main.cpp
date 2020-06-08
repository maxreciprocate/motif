#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <thread>
#include <mutex>
#include <cstring>
#include "motif.h"
#include "src/readers/file_readers.h"
#include "src/queue/ConcurrentQueue.h"

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
inline long long to_ms(const D& d) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

int main(int argc, char** argv) {
    if (argc != REQUIRED_ARGS_AMOUNT) {
        std::cerr << "Wrong arguments amount (expected 4). Usage: \n"
                     "<program> <genomes-filename> <markers-filename>"
                     "<output-filename>\n" << std::endl;
        return 1;
    }

    const uint8_t threads_num = 4;

    std::string f_genomes_path(argv[GENOMES_FILE_ARG]);

    // read genomes paths
    std::vector<std::string> genomes_paths;
    read_genome_paths(f_genomes_path, genomes_paths);

    // read markers
    std::ifstream f_markers(argv[MARKERS_FILE_ARG]);
    if (!f_markers.good()) {
        std::cerr << "Cannot read file: " << argv[MARKERS_FILE_ARG] << std::endl;
        return 1;
    }

    const MARKERS_DATA markersData = read_markers(f_markers);

    f_markers.close();
    const AUTOMATON automaton = create_automaton(markersData);

    // find matches
    ConcurrentQueue<file_entry> reading_queue;
    ConcurrentQueue<file_entry> writing_queue;

    std::thread producer([&]() {
        for (auto& file: genomes_paths) {
            file_entry new_file_entry(file);
            read_file(file, new_file_entry);
            reading_queue.push(new_file_entry);
        }
        // add poisson pills
        for (int i = 0; i < threads_num; ++i) {
            file_entry poisson_pill;
            reading_queue.push(poisson_pill);
        }
    });

    // create consumers
    std::vector<std::thread> threads;
    threads.reserve(threads_num);
    std::mutex file_write_mutex;
    std::ofstream result_file(argv[OUTPUT_FILE_ARG]);

    if (!result_file.good()) {
        std::cerr << "Cannot open file: " << argv[OUTPUT_FILE_ARG] << std::endl;
        return 1;
    }

    for (int i = 0; i < threads_num; ++i) {
        threads.emplace_back(
            [&, i]() {

                auto new_file_entry = reading_queue.pop();
                // die on poisson pill
                while (!new_file_entry.file_name.empty()) {

                    file_entry result_file_entry(new_file_entry.file_name.data());
                    result_file_entry.content.resize(markersData.markers.size());

                    std::memset(result_file_entry.content.data(), '0', markersData.markers.size());

                    match(
                        new_file_entry.content,
                        automaton.automaton,
                        automaton.output_links,
                        result_file_entry.content
                    );

                    writing_queue.push(result_file_entry);
                    new_file_entry = reading_queue.pop();
                }
                file_entry poisson_pill("");
                writing_queue.push(poisson_pill);
            }
        );
    }

    // create saver
    auto poisson_pills_counter = 0;
    std::thread saver([&]() {
        while (poisson_pills_counter != threads_num) {
            auto new_file_entry = writing_queue.pop();
            if (new_file_entry.file_name.empty()) {
                poisson_pills_counter++;
            } else {
                result_file << std::filesystem::path(new_file_entry.file_name).filename().c_str() << ' ' << new_file_entry.content << std::endl;
            }
        }
    });

    // close all threads
    producer.join();
    for (auto& th: threads)
        th.join();

    saver.join();
    result_file.close();

    return 0;
}
