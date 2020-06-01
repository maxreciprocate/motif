#include <stdint.h>

#include <string>
#include <vector>

#include "jam.h"

#define noteError(msg) \
  { noteErrorM((msg), __FILE__, __LINE__); }

inline void noteErrorM(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) fprintf(stderr, "(cuda): %s %s %d\n", cudaGetErrorString(code), file, line);
}

texture<uint32_t, cudaTextureType1D> t_table;
texture<uint8_t, cudaTextureType1D> t_translation;

__global__ void launch(char* d_source, uint32_t size, uint8_t* d_output) {
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

      if (wordidx != 0) d_output[wordidx - 1] = 0x31;
    }
  }
}

void setup(std::vector<uint32_t>& table) {
  uint8_t* d_translation;

  noteError(cudaMalloc((void**)&d_translation, Lut.size()));
  noteError(cudaMemcpy(d_translation, &Lut, Lut.size(), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_translation, d_translation, Lut.size()));

  uint32_t* d_table;

  noteError(cudaMalloc((void**)&d_table, table.size() * sizeof(uint32_t)));
  noteError(cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_table, d_table, table.size() * sizeof(uint32_t)));
}

void match(char* d_source, std::string& source, uint8_t* d_output, std::vector<uint8_t>& output, float* time) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 dimGrid(64000);
  dim3 dimBlock(1024);

  cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output.data(), output.size(), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  launch<<<dimGrid, dimBlock>>>(d_source, source.size(), d_output);

  cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(time, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
