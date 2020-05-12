#include "archive_reader.h"
#include "file_readers.h"

//template <class S>
//int read_files_from_archive(const std::string& archive_name, std::vector<file_entry>& container) {
//    std::string archive_content;
//    read_file(archive_name, archive_content);
//
//    struct archive *a;
//    struct archive_entry *entry;
//    int r;
//    a = archive_read_new();
//    if (!a)
//        return -1;
//
//    if (archive_read_support_filter_all(a) != ARCHIVE_OK)
//        return -1;
//
//    if (archive_read_support_format_all(a) != ARCHIVE_OK)
//        return -1;
//
//    r = archive_read_open_memory(a, archive_content.data(), archive_content.size());
//
//    if (r != ARCHIVE_OK)
//        return -1;
//
//    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
//        if (archive_entry_filetype(entry) == AE_IFREG) {
//            size_t file_size = archive_entry_size(entry);
//
//            file_entry& file = container[0];
//
//            file.file_name = archive_entry_pathname(entry);
//            file.content.resize(file_size);
//
//            la_ssize_t bytes_read = archive_read_data(a, file.content.data(), file_size);
//
//            if (bytes_read != file_size) {
//                return -1;
//            }
//
//            break;
//        }
//    }
//
//    return archive_read_free(a);
//}
