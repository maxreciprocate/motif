#include <string>
#include <vector>
#include <array>

#ifndef JAM_H
#define JAM_H

/// 0x00 or 0xff for filling is equally enough
/// Some out-of-bounds measures have to be taken in case genome has some bit flips
const std::array<uint8_t, 24> Lut = {
  0x0, 0x1, 0x0, 0x2, 0x0, 0x0, 0x0, 0x3,
  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  0x0, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0,
};

void match(char* d_source, std::string& source, uint8_t *d_output, std::vector<uint8_t>& output, float* time);

void setup(std::vector<uint32_t>& table);

#endif
