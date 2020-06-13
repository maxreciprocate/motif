#include <iostream>

#include <string>
#include <vector>

#include "jam.h"

texture<uint32_t, cudaTextureType1D> t_table;
texture<uint8_t, cudaTextureType1D> t_translation;
cudaChannelFormatDesc uint8Desc = cudaCreateChannelDesc<uint8_t>();
cudaChannelFormatDesc uint32Desc = cudaCreateChannelDesc<uint32_t>();

#define N_CODE    5

__device__ void decode_3_char(char encoded_char, uint8_t data[3]) {
  
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

__device__ void p_c(char a) {
    for (int i = 0; i < 8; i++) {
      printf("%d", !!((a << i) & 0x80));
    }
    printf("\n");
}


__host__ void p_c_h(char a) {
    for (int i = 0; i < 8; i++) {
      printf("%d", !!((a << i) & 0x80)); 
    }
    printf("\n");
}

__global__ void launch(char* d_source, uint32_t size, int8_t* d_output) {
  uint32_t stride = gridDim.x * blockDim.x;

  uint8_t data[3];

  for (uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x; tidx < size; tidx += stride) {
    uint32_t vx = 0;

    
    for (uint32_t idx = tidx; idx < size; ++idx) {
      char encoded_char = d_source[idx / 3];
      // p_c(encoded_char);
      decode_3_char(encoded_char, data);

      //printf("data: %d, %d, %d\n", data[0], data[1], data[2]);
      
      uint8_t data_index = idx % 3;

      //printf("tidx: %d idx: %d data_index: %d\n", tidx, idx, data_index);      
        uint8_t offset = data[data_index];

        if (offset == N_CODE) break;

        vx = tex1Dfetch(t_table, vx * 5 + offset);

        //printf("tidx: %d vx: %d\n", tidx, vx);
        if (vx == 0) break;

        uint32_t wordidx = tex1Dfetch(t_table, vx * 5 + 4);

        //printf("tidx: %d wordidx: %d\n", tidx, wordidx);
        
        if (wordidx != 0) d_output[wordidx - 1] = 1;
      
      if (data_index == 2) continue;
      if (++idx == size) break;

        offset = data[data_index + 1];

        if (offset == N_CODE) break;

        vx = tex1Dfetch(t_table, vx * 5 + offset);

        //printf("tidx: %d vx: %d\n", tidx, vx);
        if (vx == 0) break;

        wordidx = tex1Dfetch(t_table, vx * 5 + 4);

        //printf("wordidx: %d\n", wordidx);

        if (wordidx != 0) d_output[wordidx - 1] = 1;
      

      //printf("tidx: %d data_index: %d before break\n", tidx, data_index);

      if (data_index == 1) continue;
      if (++idx == size) break;

      //printf("tidx: %d data_index: %d after break\n", tidx, data_index);

        offset = data[data_index + 2];

        if (offset == N_CODE) break;

        vx = tex1Dfetch(t_table, vx * 5 + offset);

        //printf("tidx: %d vx: %d\n", tidx, vx);
        if (vx == 0) break;

        wordidx = tex1Dfetch(t_table, vx * 5 + 4);

        //printf("wordidx: %d\n", wordidx);

        if (wordidx != 0) d_output[wordidx - 1] = 1;
      
    }
  }
}

void setup(uint32_t*& d_table, std::vector<uint32_t>& table) {
  uint8_t* d_translation;

  noteError(cudaMalloc((void**)&d_translation, Lut.size()));
  noteError(cudaMemcpy(d_translation, &Lut, Lut.size(), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_translation, d_translation, uint8Desc, Lut.size()));

  noteError(cudaMalloc((void**)&d_table, table.size() * sizeof(uint32_t)));
  noteError(cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  //for (int row = 0; row < (table.size() / 5); ++row) {
  //  for (int col = 0; col < 5; ++col) {
  //    printf("%d ", table[row * 5 + col]);
  //  }
  //  printf("\n");
  //}


  noteError(cudaBindTexture(0, t_table, d_table, uint32Desc, table.size() * sizeof(uint32_t)));
}

void match(char* d_source, encodedGenomeData& source, int8_t* d_output, int8_t* output, int64_t output_size) {
  //dim3 dimGrid(1);
  //dim3 dimBlock(source.data.size());
  dim3 dimGrid(std::max(source.data.size() >> 11, static_cast<size_t>(32768)));
  dim3 dimBlock(1024);


  //std::cout << source.data.size() << std::endl;
  //p_c_h(source.data[0]);
  //std::cout << source.real_size << std::endl;
  noteError(cudaMemcpy(d_source, source.data.data(), source.data.size(), cudaMemcpyHostToDevice));
  noteError(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice));

  launch<<<dimGrid, dimBlock>>>(d_source, source.real_size, d_output);

  noteError(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
}

void clear_table(uint32_t* d_table) {
  noteError(cudaUnbindTexture(t_table));
  noteError(cudaFree(d_table));
}
