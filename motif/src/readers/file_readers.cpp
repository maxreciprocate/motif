#include <sstream>
#include "file_readers.h"

MARKERS_DATA read_markers(std::ifstream &in) {
    uint64_t sum_of_all_chars = 0;
    uint32_t longest_marker_len = 0;
    std::string line;

    MARKERS_DATA markersData{};
    while (std::getline(in, line)) {
        markersData.markers.emplace_back(
            line.begin() + line.find(',') + 1, line.end()
        );

        longest_marker_len = std::max(longest_marker_len, static_cast<uint32_t>(line.size()));
        sum_of_all_chars += line.size();
    }

    markersData.sum_of_all_chars = sum_of_all_chars;
    markersData.longest_marker_len = longest_marker_len;

    return markersData;
}

void read_genome_paths(const std::string &file_name, std::vector<std::string> &container) {
    std::string container_str;
    read_file(file_name, container_str);

    const char new_line_delimiter = '\n';
    auto lines_number = std::count(container_str.begin(), container_str.end(), new_line_delimiter);

    container.reserve(lines_number + 1);

    int64_t from = -1;
    int64_t to;
    std::string path(
        std::filesystem::path(file_name).remove_filename().string()
    );

    for (auto i = 0; i < lines_number; ++i) {
        to = container_str.find_first_of(new_line_delimiter, ++from);


        std::string genome_path(container_str, from, to - from);

        if (genome_path.empty()) continue;

        genome_path = path + genome_path;

        container.emplace_back(
            std::move(genome_path)
        );

        from = to;
    }

    ++from;
    std::string genome_path(container_str, from, container_str.size() - from);
    if (!genome_path.empty()) {
        genome_path = path + genome_path;

        container.emplace_back(
            std::move(genome_path)
        );
    }
}


auto ignore_file(std::ifstream &in, const std::string_view &file_name) {
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

void read_file(const std::string &file_name, std::string &container) {
    std::ifstream in(file_name, std::ifstream::binary);

    if (!in.good()) {
        std::ostringstream msg;
        msg << "Cannot read file: " << file_name;
        throw std::runtime_error(msg.str());
    }
//  auto const char_count = ignore_file(in, file_name);
    // get length of file:
    in.seekg(0, in.end);
    auto const char_count = in.tellg();
    in.seekg(0, in.beg);

    container.resize(char_count);

    if (!in.read(container.data(), char_count)) {
        throw std::runtime_error("Can't read file");
    }

    in.close();
}

void read_file(const std::string &file_name, std::string_view &container) {
    std::ifstream in(file_name, std::ifstream::binary);

    if (!in.good()) {
        std::ostringstream msg;
        msg << "Cannot read file: " << file_name;
        throw std::runtime_error(msg.str());
    }
    auto const char_count = ignore_file(in, file_name);
//    // get length of file:
//    in.seekg(0, in.end);
//    auto const char_count = in.tellg();
//    in.seekg(0, in.beg);

    std::unique_ptr<char> container_pointer(new char[char_count]);

    if (!in.read(container_pointer.get(), char_count)) {
        throw std::runtime_error("Can't read file");
    }

    container = std::string_view(container_pointer.get(), char_count);
    in.close();
}

void read_file(const std::string &file_name, file_entry &container) {
    read_file(file_name, container.content);
//    container.file_name = file_name;
}
