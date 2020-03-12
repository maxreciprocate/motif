#include <string_view>
#include <vector>

#ifndef MOTIF
#define MOTIF


int ooo();

std::string_view match(const std::string& source,
    const std::vector<std::string>& markers);

#endif
