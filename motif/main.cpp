#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <thread>
#include <mutex>
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
    std::vector<std::string> genomes_paths;
    read_genome_paths(f_genomes_path, genomes_paths);

//    std::cout << genomes_paths[8] << std::endl;

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


    std::vector<uint32_t> automaton((max_states_num + 1) << MATRIX_WIDTH_LEFT_SHIFT, 0);

    std::vector<std::vector<uint32_t>> output_links(max_states_num);

    create_automaton(markers, automaton, output_links);

    end = get_current_time_fenced();

    std::cout << "Automaton created: " << to_us(end - start) << std::endl;


    // find matches
    ConcurrentQueue<file_entry> reading_queue;
    ConcurrentQueue<file_entry> writing_queue;

    std::thread producer([&]() {
        for (auto& file: genomes_paths) {
            if (file.empty()) continue;
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
                    result_file_entry.content.resize(markers.size());

                    std::memset(result_file_entry.content.data(), '0', markers.size());

                    match(
                        new_file_entry.content,
                        automaton,
                        output_links,
                        result_file_entry.content
                    );


                    writing_queue.push(result_file_entry);

                    std::cout << " analyzed " << new_file_entry.file_name << std::endl;

                    new_file_entry = reading_queue.pop();
                }
                file_entry poisson_pill("");
                writing_queue.push(poisson_pill);
                std::cout << "thread " << i << " is closing " << std::endl;
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
                std::cout << "Saved " << new_file_entry.file_name << std::endl;
                result_file << std::filesystem::path(new_file_entry.file_name).filename().c_str() << ' ' << new_file_entry.content << std::endl;
            }
        }
    });
    // close all threads
    producer.join();
    for (auto& th: threads) {
        th.join();
    }
    saver.join();
    result_file.close();

    end = get_current_time_fenced();

    std::cout << "Genomes analized: " << to_us(end - start) << std::endl;
    return 0;
}
