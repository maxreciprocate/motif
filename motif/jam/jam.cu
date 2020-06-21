#include <iostream>

#include <string>
#include <vector>
#include <chrono>
#include <atomic>

#include "jam.h"

texture<uint32_t, cudaTextureType1D> t_table;
texture<uint8_t, cudaTextureType1D> t_translation;
cudaChannelFormatDesc uint8Desc = cudaCreateChannelDesc<uint8_t>();
cudaChannelFormatDesc uint32Desc = cudaCreateChannelDesc<uint32_t>();

#define N_CODE    5

__device__ void decode_3_char(char encoded_char, volatile uint8_t* data) {
  
  uint8_t countN = encoded_char & 0b11;
    
   if (countN == 0) {
    // decode all 3 chars by 2 bits
    encoded_char >>= 2;
    data[0] = encoded_char & 0b11;

    encoded_char >>= 2;
    data[1] = encoded_char & 0b11;

    encoded_char >>= 2;
    data[2] = encoded_char & 0b11;
  } else if (countN == 1) {
    // in this case we also need to get index of N 
    encoded_char >>= 2;
    uint8_t index = encoded_char & 0b11;
    data[index] = N_CODE;
    
    // decode 2 "not N" chars
    for (uint8_t i = 0; i < 3; ++i) {
        if (i == index) continue;
        encoded_char >>= 2;
        data[i] = encoded_char & 0b11;
    }
  } else if (countN == 2) {
    // in these case ("N_N", since we replace "NN" by "N") decode only middle char 
    encoded_char >>= 2;
    data[0] = N_CODE;
    data[1] = encoded_char & 0b11;
    data[2] = N_CODE;
  } else { // countN == 3
    // will be created in future version of the algorithm, when we will analyze 'N'
  }

  
}

#define NUM_CHARS_IN_BYTE  3
#define BLOCK_DIM          1024

__global__ void launch(char* d_source, uint32_t size, int8_t* d_output) {
  __shared__ uint8_t s_source[BLOCK_DIM * NUM_CHARS_IN_BYTE];

  uint32_t stride = gridDim.x * blockDim.x;

  for (size_t tidx = blockIdx.x * blockDim.x + threadIdx.x; tidx < size; tidx += stride) {
    uint32_t bidx = tidx - threadIdx.x;
    uint32_t vx = 0;

    //if (threadIdx.x < 580) {
      decode_3_char(
        d_source[(bidx / 3) + threadIdx.x],
        &s_source[threadIdx.x * 3]
      );
    //}

    __syncthreads();

    uint16_t s_source_index = threadIdx.x + (bidx % 3);
    for (uint32_t global_index = tidx; global_index < size; ++global_index) {

      uint8_t offset = s_source[s_source_index++];

      if (offset == N_CODE) break;

      vx = tex1Dfetch(t_table, vx * 5 + offset);

      if (vx == 0) break;

      uint32_t wordidx = tex1Dfetch(t_table, vx * 5 + 4);

      if (wordidx != 0) d_output[wordidx - 1] = 1;

    }
  
    __syncthreads();
  }
}

void setup(uint32_t*& d_table, std::vector<uint32_t>& table) {
  uint8_t* d_translation;

  noteError(cudaMalloc((void**)&d_translation, Lut.size()));
  noteError(cudaMemcpy(d_translation, &Lut, Lut.size(), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_translation, d_translation, uint8Desc, Lut.size()));

  noteError(cudaMalloc((void**)&d_table, table.size() * sizeof(uint32_t)));
  noteError(cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  noteError(cudaBindTexture(0, t_table, d_table, uint32Desc, table.size() * sizeof(uint32_t)));
}

void match(char* d_source, encodedGenomeData& source, int8_t* d_output, int8_t* output, int64_t output_size) {
  dim3 dimGrid(std::max(source.data.size() >> 11, static_cast<size_t>(32768)));
  dim3 dimBlock(1024);

  noteError(cudaMemcpy(d_source, source.data.data(), source.data.size(), cudaMemcpyHostToDevice));
  noteError(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice));

  launch<<<dimGrid, dimBlock>>>(d_source, source.real_size, d_output);

  noteError(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
}

void clear_table(uint32_t* d_table) {
  noteError(cudaUnbindTexture(t_table));
  noteError(cudaFree(d_table));
}
