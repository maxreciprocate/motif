#include <stdint.h>
#include <algorithm>

#include <string>
#include <vector>

#include "jam.h"

#define ARRAY_WIDTH_MAX        ((1 << 16) - 1)
#define ARRAY_WIDTH_MAX_SHIFT  16

#define noteError(msg) \
  { noteErrorM((msg), __FILE__, __LINE__); }

inline void noteErrorM(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) fprintf(stderr, "(cuda): %s %s %d\n", cudaGetErrorString(code), file, line);
}

texture<uint32_t, cudaTextureType1D> t_table;
texture<uint8_t, cudaTextureType1D> t_translation;

__device__ void setMarker(int index, cudaSurfaceObject_t resultSurfObj) {
  int col = index % ARRAY_WIDTH_MAX;
  int row = index / ARRAY_WIDTH_MAX;
  //if (col > 65530) printf("col: %d, row: %d\n", col, row);
  surf2Dwrite<uint8_t>(0x31, resultSurfObj, col, row);
}

__global__ void launch(char* d_source, uint32_t size, cudaSurfaceObject_t resultSurfObj) {

  //surf2Dwrite<uint8_t>(0x31, resultSurfObj, 65534 * 2, 8);
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

      if (wordidx != 0) {
        //d_output[wordidx - 1] = 0x31;
        setMarker(wordidx - 1, resultSurfObj);
        //surf1Dwrite<uint8_t>(0x31, resultSurfObj, wordidx - 1);
      }
    }
  }
}

void setup(std::vector<uint32_t>& table) {
  uint8_t* d_translation;

  noteError(cudaMalloc((void**)&d_translation, Lut.size()));
  noteError(cudaMemcpy(d_translation, &Lut, Lut.size(), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_translation, d_translation));

  uint32_t* d_table;

  noteError(cudaMalloc((void**)&d_table, table.size() * sizeof(uint32_t)));
  noteError(cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  noteError(cudaBindTexture(0, t_table, d_table));
}

void match(char* d_source, std::string& source, uint8_t* d_output, std::vector<uint8_t>& output, float* time) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 dimGrid(64000);
  dim3 dimBlock(1024);

  int index = 655360;
  printf("test div: %d, res: %d\n", index >> 16, index & ARRAY_WIDTH_MAX);
  cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output.data(), output.size(), cudaMemcpyHostToDevice);

  // Allocate CUDA arrays in device memory
  cudaChannelFormatDesc doubleDesc = cudaCreateChannelDesc<uint8_t>();
  cudaArray* cuResultArray;

  //printf("ARRAY_WIDTH_MAX: %d/n", ARRAY_WIDTH_MAX);
  //printf("output.size(): %lu\n\n", output.size());

  const size_t resultWidth = ARRAY_WIDTH_MAX;
  const size_t resultHeight = (output.size() >> 16) + ((output.size() & ARRAY_WIDTH_MAX) && 1);

  noteError(cudaMallocArray(&cuResultArray, &doubleDesc, resultWidth, resultHeight,
                  cudaArraySurfaceLoadStore));
  int val;
  cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DHeight, 0);
  //printf("cudaDevAttrMaxSurface1D: %d\n", val);

  // Specify surface
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

  // Create the surface objects
  resDesc.res.array.array = cuResultArray;
  cudaSurfaceObject_t resultSurfObj = 0;
  cudaCreateSurfaceObject(&resultSurfObj, &resDesc);

  cudaEventRecord(start, 0);
  launch<<<dimGrid, dimBlock>>>(d_source, source.size(), resultSurfObj);

  //printf("test: %d/n", ((int) output.size()) % ARRAY_WIDTH_MAX);
  //cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost);
  //cudaMemcpy(output.data(), cuResultArray, output.size(), cudaMemcpyDeviceToHost);
  //printf("resultHeight: %d\n", resultHeight);
  noteError(cudaMemcpy2DFromArray(output.data(), resultWidth, cuResultArray, 0, 0, resultWidth, resultHeight - 1, cudaMemcpyDeviceToHost));
  noteError(cudaMemcpy2DFromArray(output.data() + (resultWidth * (resultHeight - 1)), resultWidth, cuResultArray, 0, resultHeight - 1, (output.size() % ARRAY_WIDTH_MAX) + 1, 1, cudaMemcpyDeviceToHost));

std::replace(output.begin(), output.end(), 0, 0x30);
  //for (int i = 0; i < 20; ++i) {
  //  printf("%d", output.data()[i]);
  //}
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(time, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  cudaFreeArray(cuResultArray);
}
