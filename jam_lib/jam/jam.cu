#include <stdint.h>
#include <vector>
#include <string>
#include <mutex>

__global__
void launch(uint32_t *d_table, char* d_source, uint32_t size, uint8_t *d_lut, int8_t *d_output) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  while (tidx < size) {
    char c = d_source[tidx];

    if (c == 0x4e) {
      tidx += stride;
      continue;
    }

    int vx = d_table[d_lut[c] - 1];
    int idx = tidx;

    while (true) {
      int wordidx = d_table[5 * vx + 4];

      if (wordidx != 0)
        d_output[wordidx - 1] = 1;

      idx += 1;
      if (idx > size || vx == 0)
        break;

      c = d_source[idx];
      if (c == 0x4e || c < 0x41)
        break;

      vx = d_table[5 * vx + d_lut[c] - 1];
    }

    tidx += stride;
  }
}

void match(uint32_t *d_table, char* d_source, uint32_t size, uint8_t *d_lut,
           int8_t *d_output, int8_t* output, int64_t output_size, std::string& source) 
{
  cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice);

  launch<<<8000, 1024>>>(d_table, d_source, source.size(), d_lut, d_output);
  
  cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
}
