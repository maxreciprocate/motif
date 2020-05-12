#ifndef READING_WORDS_ARCHIVE_READER_H
#define READING_WORDS_ARCHIVE_READER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
//#include <archive.h>
//#include <archive_entry.h>

struct file_entry {
    std::string file_name;
    std::string content;

    file_entry() = default;

    explicit file_entry(const char *new_file_name) {
        file_name = new_file_name;
    }
    explicit file_entry(const std::string& new_file_name) {
        file_name = new_file_name;
    }
    file_entry(const char *new_file_name, const std::string_view& new_content) {
        file_name = new_file_name;
        content = new_content;
    }

    file_entry(const char *new_file_name, const std::string& new_content) {
        file_name = new_file_name;
        content = new_content;
    }


};

int read_files_from_archive(const std::string& archive_name, std::vector<file_entry>& container);

#endif //READING_WORDS_ARCHIVE_READER_H
