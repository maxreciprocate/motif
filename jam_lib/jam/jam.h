#include <string>
#include <vector>

#ifndef JAM_H
#define JAM_H

void match(uint32_t *d_table, char* d_source, uint32_t size, uint8_t *d_lut, 
          int8_t *d_output, std::vector<int8_t>& output, std::string& source);

#endif
