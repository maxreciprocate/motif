#include <string>
#include <vector>
#include <array>

#ifndef JAM_H
#define JAM_H

#define debug 0

#define noteError(msg) \
  { noteErrorM((msg), __FILE__, __LINE__); }

inline void noteErrorM(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "(cuda error): %s %s %d\n", cudaGetErrorString(code), file, line);
    if (!debug) exit(1);
  }
}

const std::array<uint8_t, 24> Lut = {
  0x0, 0x1, 0x0, 0x2, 0x0, 0x0, 0x0, 0x3,
  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  0x0, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0,
};

typedef struct {
  std::string data;
  size_t real_size;
} encodedGenomeData;

void match(char* d_source, encodedGenomeData& source, int8_t* d_output, int8_t* output, int64_t output_size);
void setup(uint32_t*& d_table, std::vector<uint32_t>& table);
void clear_table(uint32_t* d_table);

#endif
