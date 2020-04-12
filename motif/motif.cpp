#include "motif.h"
#include "src/readers/file_readers.h"

uint8_t get_index_to_search(
    std::vector<uint32_t> &matrix,
    uint32_t parent_state,
    uint32_t child_state
) {

    for (uint8_t index = 0; index < AUTOMATON_WIDTH; ++index) {

        if (matrix[(parent_state << MATRIX_WIDTH_LEFT_SHIFT) + index] == child_state) {
            return index;
        }
    }

    // if this error was thrown than code is not working at all
    throw std::runtime_error("Huge problem");
}

uint64_t create_goto(
    const MARKERS_DATA &markersData,
    AUTOMATON &matrix_goto
) {
    auto& markers = markersData.markers;
    auto log_4_markers_size = std::log(markers.size()) / std::log(4);

    uint32_t max_states_num = (
        (std::pow(4, std::ceil(log_4_markers_size)) - 1) / (4 - 1)
    ) + markersData.sum_of_all_chars - std::floor(log_4_markers_size) * markers.size() + 3;

    std::array<uint8_t, static_cast<uint8_t>('A') + 1 + ALPHABET_SIZE> to_index{};

    to_index['A'] = 0;
    to_index['T'] = 1;
    to_index['G'] = 2;
    to_index['C'] = 3;


    // create goto
    uint32_t cur_state, next_states = 2;
    uint32_t marker_index = 0;
    std::vector<uint32_t> matrix((max_states_num + 1) << MATRIX_WIDTH_LEFT_SHIFT, 0);
    std::vector<std::vector<uint32_t>> output_links(max_states_num);
    for (auto it = markers.cbegin(); it != markers.cend(); ++it, ++marker_index) {
        cur_state = START_STATE;

        auto &marker = *it;

        for (auto chr: marker) {
            if (!matrix[(cur_state << MATRIX_WIDTH_LEFT_SHIFT) + to_index[chr]]) {
                matrix[(cur_state << MATRIX_WIDTH_LEFT_SHIFT) + to_index[chr]] = next_states++;
            }
            cur_state = matrix[(cur_state << MATRIX_WIDTH_LEFT_SHIFT) + to_index[chr]];
        }

        if (output_links[cur_state].empty()) {
            output_links[cur_state] = {marker_index};
        } else {
            output_links[cur_state].push_back(marker_index);
        }

    }

    matrix_goto.automaton = matrix;
    matrix_goto.output_links = output_links;

    return next_states;
}

AUTOMATON create_double_build(
    const AUTOMATON &matrix_goto,
    const uint64_t num_states
) {

    std::vector<uint32_t> matrix(num_states << MATRIX_WIDTH_LEFT_SHIFT, 0);
    std::vector<uint32_t> queue;
    queue.reserve(num_states);

    auto& matrix__goto = matrix_goto.automaton;

    queue[START_STATE] = START_STATE;
    uint32_t current_state_index = 2;
    for (uint32_t state_index = START_STATE; state_index < num_states; ++state_index) {
        auto current_state = queue[state_index];

        for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
            auto child_state = matrix__goto[(current_state << MATRIX_WIDTH_LEFT_SHIFT) + state];
            if (child_state) {
                matrix[(state_index << MATRIX_WIDTH_LEFT_SHIFT) + state] = current_state_index;
                queue[current_state_index++] = child_state;
            }
        }


    }

    // set links to root node for root node's children
    for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
        if (!matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state]) {
            matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state] = START_STATE;
        }
    }

    // update output links
    std::vector<std::vector<uint32_t>> output_links(num_states);

    auto& goto_output_links = matrix_goto.output_links;
    for (uint32_t j = START_STATE; j < num_states; ++j) {
        output_links[j] = goto_output_links[queue[j]];
    }

    return {
        .automaton = matrix,
        .output_links = output_links
    };
}

AUTOMATON create_automaton(
    const MARKERS_DATA &markersData
) {

    // create goto
    AUTOMATON matrix_goto{};
    const uint64_t num_states = create_goto(markersData, matrix_goto);

    // double-build
    AUTOMATON automaton = create_double_build(matrix_goto, num_states);

    auto& matrix = automaton.automaton;
    auto& output_links = automaton.output_links;

    // create fail links
    std::vector<uint32_t> fail_links(num_states, 0);
    std::queue<uint32_t> states_queue;

    for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
        if (matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state] > START_STATE) {
            auto curr_state = matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state];
            fail_links[curr_state] = START_STATE;
            states_queue.push(curr_state);
        }
    }


    uint32_t fail_parent_state, parent_state, child_state, fail_link;
    uint8_t index_to_search = 0;
    while (!states_queue.empty()) {
        parent_state = states_queue.front();
        states_queue.pop();

        for (uint8_t index = 0; index < AUTOMATON_WIDTH; ++index) {
            child_state = matrix[(parent_state << MATRIX_WIDTH_LEFT_SHIFT) + index];

            if (child_state) {

                states_queue.push(child_state);


                // save fail link
                fail_parent_state = fail_links[parent_state];
                index_to_search = get_index_to_search(
                    matrix, parent_state, child_state
                );

                while (true) {

                    fail_link = matrix[(fail_parent_state << MATRIX_WIDTH_LEFT_SHIFT) + index_to_search];
                    if (fail_link) {

                        fail_links[child_state] = fail_link;

                        if (output_links[child_state].empty()) {
                            // save just the reference to the same vector since there is no need in copying
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
        if (matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state] > START_STATE) {
            states_queue.push(matrix[(START_STATE << MATRIX_WIDTH_LEFT_SHIFT) + state]);
        }
    }

    while (!states_queue.empty()) {
        parent_state = states_queue.front();
        states_queue.pop();


        for (uint8_t state = 0; state < AUTOMATON_WIDTH; ++state) {
            if (matrix[(parent_state << MATRIX_WIDTH_LEFT_SHIFT) + state]) {
                states_queue.push(matrix[(parent_state << MATRIX_WIDTH_LEFT_SHIFT) + state]);
            } else {
                matrix[(parent_state << MATRIX_WIDTH_LEFT_SHIFT) + state] =
                    matrix[(fail_links[parent_state] << MATRIX_WIDTH_LEFT_SHIFT) + state];
            }
        }
    }

    return automaton;
}


void match(
    const std::string_view &source,
    const std::vector<uint32_t> &automaton,
    const std::vector<std::vector<uint32_t>> &output_links,
    std::string &result
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

        current_state = automaton[(current_state << MATRIX_WIDTH_LEFT_SHIFT) + to_index[chr]];

        for (auto &output: output_links[current_state]) {
            result[output] = '1';
        }
    }
}

//void match_genome(
//    const file_entry &genome,
//    uint64_t markers_size,
//    const std::vector<uint32_t> &automaton,
//    const std::vector<std::vector<uint32_t>> &output_links,
//    std::mutex &file_write_mutex,
//    std::string& result,
//    std::ofstream &output_file
//) {
//    std::memset(result.data(), '0', markers_size);
//
//    match(genome.content, automaton, output_links, result);
//
//    {
//        std::lock_guard<std::mutex> lg{file_write_mutex};
//        output_file << std::filesystem::path(genome.file_name).filename().c_str() << ' ' << result << std::endl;
//    }
//}