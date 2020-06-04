#include <stdint.h>

#include <string>
#include <vector>

#include "jam.h"

#define noteError(msg) \
  { noteErrorM((msg), __FILE__, __LINE__); }

inline void noteErrorM(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "(cuda error): %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

texture<uint32_t, cudaTextureType1D> t_table;
texture<uint8_t, cudaTextureType1D> t_translation;

__global__ void launch(char* d_source, uint32_t size, int8_t* d_output) {
  uint32_t stride = gridDim.x * blockDim.x;

  for (uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x; tidx < size; tidx += stride) {
    uint32_t vx = 0;

    for (uint32_t idx = tidx; idx < size; ++idx) {
      uint8_t c = d_source[idx];

      if (c == 0x4e) break;

      uint32_t offset = tex1Dfetch(t_translation, c - 0x40);
      vx = tex1Dfetch(t_table, 5 * vx + offset - 1);

      if (vx == 0) break;

      uint32_t wordidx = tex1Dfetch(t_table, 5 * vx + 4);

      if (wordidx != 0) d_output[wordidx - 1] = 1;
    }
  }
}

void setup(uint32_t* d_table, std::vector<uint32_t>& table) {
  uint8_t* d_translation;

  noteError(cudaMalloc((void**)&d_translation, Lut.size()));
  noteError(cudaMemcpy(d_translation, &Lut, Lut.size(), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_translation, d_translation));

  noteError(cudaMalloc((void**)&d_table, table.size() * sizeof(uint32_t)));
  noteError(cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  noteError(cudaBindTexture(0, t_table, d_table));
}

void match(char* d_source, std::string& source, int8_t* d_output, int8_t* output, int64_t output_size) {
  dim3 dimGrid(std::max(source.size() >> 11, static_cast<size_t>(32768)));
  dim3 dimBlock(1024);

  noteError(cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice));
  noteError(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice));

  launch<<<dimGrid, dimBlock>>>(d_source, source.size(), d_output);

  noteError(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
}
