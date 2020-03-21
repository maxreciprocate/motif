#include "motif.h"
#include "file_readers.h"

uint8_t get_index_to_search(
    std::vector<std::array<uint32_t, MATRIX_WIDTH>> &matrix,
    uint32_t parent_state,
    uint32_t child_state
) {
    for (uint8_t index = 0; index < AUTOMATON_WIDTH; ++index) {

        if (matrix[parent_state][index] == child_state) {
            return index;
        }
    }

    return MATRIX_WIDTH + 1; // in fact will never return this value
}


void create_automaton(
    const std::deque<std::string> &markers,
    std::vector<std::array<uint32_t, MATRIX_WIDTH>> &matrix,
    std::vector<std::vector<uint32_t>> &output_links
) {
    std::array<uint8_t, static_cast<uint8_t>('A') + 1 + ALPHABET_SIZE> to_index{};

    to_index['A'] = 0;
    to_index['T'] = 1;
    to_index['G'] = 2;
    to_index['C'] = 3;

    std::vector<uint32_t> fail_links(matrix.size(), 0);

    // create goto
    uint32_t cur_state, next_states = 2;
    uint32_t marker_index = 0;
    for (auto it = markers.cbegin(); it != markers.cend(); ++it, ++marker_index) {
        cur_state = START_STATE;

        auto& marker = *it;

        for (auto chr: marker) {
            if (!matrix[cur_state][to_index[chr]]) {
                matrix[cur_state][to_index[chr]] = next_states;

                next_states++;
            }
            cur_state = matrix[cur_state][to_index[chr]];
        }

        if (output_links[cur_state].empty()) {
            output_links[cur_state] = {marker_index};
        } else {
            output_links[cur_state].push_back(marker_index);
        }

    }


    for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
        if (!matrix[START_STATE][state]) {
            matrix[START_STATE][state] = START_STATE;
        }
    }


    // create fail links
    std::queue<uint32_t> states_queue;

    for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
        if (matrix[START_STATE][state] > START_STATE) {
            fail_links[matrix[START_STATE][state]] = START_STATE;
            states_queue.push(matrix[START_STATE][state]);
        }
    }


    uint32_t fail_parent_state, parent_state, child_state;
    uint8_t index_to_search = 0;
    while (!states_queue.empty()) {
        parent_state = states_queue.front();
        states_queue.pop();

        for (uint8_t index = 0; index < AUTOMATON_WIDTH; ++index) {
            if (matrix[parent_state][index]) {
                child_state = matrix[parent_state][index];

                states_queue.push(child_state);


                // save fail link
                fail_parent_state = fail_links[parent_state];
                index_to_search = get_index_to_search(
                    matrix, parent_state, child_state
                );


                while (true) {

                    if (matrix[fail_parent_state][index_to_search]) {
                        uint32_t fail_link = matrix[fail_parent_state][index_to_search];

                        fail_links[child_state] = fail_link;

                        if (output_links[child_state].empty()) {
                            // TODO: change vector<vector<uint32_t>> to vector<vector<uint32_t>*> and
                            //       here save just pointer to output_links[fail_link]
                            output_links[child_state] = output_links[fail_link];
                        } else if (!output_links[fail_link].empty()) {
                            // copy fail vector
                            output_links[child_state].insert(
                                output_links[child_state].end(),
                                output_links[fail_link].begin(),
                                output_links[fail_link].end()
                            );
                        }
                        break;
                    }

                    if (fail_parent_state == START_STATE) {
                        fail_links[child_state] = START_STATE;
                        break;
                    }

                    fail_parent_state = fail_links[fail_parent_state];
                }
            }
        }
    }

    // create automaton

    for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
        if (matrix[START_STATE][state] > START_STATE) {
            states_queue.push(matrix[START_STATE][state]);
        }
    }

    while (!states_queue.empty()) {
        parent_state = states_queue.front();
        states_queue.pop();



        for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
            if (matrix[parent_state][state]) {
                states_queue.push(matrix[parent_state][state]);
            } else {
                matrix[parent_state][state] = matrix[fail_links[parent_state]][state];
            }
        }
    }
}


void match(
    const std::string &source,
    const std::vector<std::array<uint32_t, MATRIX_WIDTH>>& automaton,
    const std::vector<std::vector<uint32_t>>& output_links,
    std::string& result
) {

    std::vector<int8_t> to_index(static_cast<uint8_t>('A') + 1 + ALPHABET_SIZE, -1);

    to_index['A'] = 0;
    to_index['T'] = 1;
    to_index['G'] = 2;
    to_index['C'] = 3;



    uint32_t current_state = START_STATE;

    for (auto chr: source) {
        if (chr == 'N') {
            current_state = START_STATE;
            continue;
        }

        current_state = automaton[current_state][to_index[chr]];

        for (auto& output: output_links[current_state]) {
            result[output] = '1';
        }
    }
}

void match_genomes (
    const std::deque<std::string> &genome_paths,
    const std::string& f_genomes_path,
    uint64_t markers_size,
    const std::vector<std::array<uint32_t, MATRIX_WIDTH>>& automaton,
    const std::vector<std::vector<uint32_t>>& output_links,
    std::mutex& file_write_mutex,
    std::ofstream& output_file
) {
    std::string genome;
    std::string result(markers_size, '0');

    std::string path(f_genomes_path.substr(0, f_genomes_path.find_last_of('/') + 1));
    for (const auto& genome_path: genome_paths) {
        for (uint32_t i = 0; i < markers_size; ++i) {
            result[i] = '0';
        }

        read_genome_file(path + genome_path, genome);

        match(genome, automaton, output_links, result);

        {
            std::lock_guard<std::mutex> lg{file_write_mutex};
            output_file << genome_path.substr(genome_path.find_last_of('/') + 1, genome_path.size()) << ' ' << result << std::endl;
        }


    }
}