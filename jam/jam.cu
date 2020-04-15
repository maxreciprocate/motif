#include "stdio.h"
#include "string.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define debug

struct measurer {
  std::string subject;
  std::chrono::high_resolution_clock::time_point start;

  measurer(const std::string &name)
      : subject(name), start(std::chrono::high_resolution_clock::now()) {}
  ~measurer() {
    auto difference = std::chrono::high_resolution_clock::now() - start;
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(difference)
            .count();

    printf("%s: %.3f\n", subject.c_str(), static_cast<double>(duration) / 1000);
  }
};

#ifdef debug
#define measure(name) measurer __measurer(name);
#else
#define measure(name) {};
#endif

inline void readfile(const std::string &filename, std::string &container) {
  std::ifstream file(filename, std::ifstream::binary);

  if (!file.good())
    throw std::runtime_error("cannot open file");

  file.seekg(0, file.end);
  const auto nchars = file.tellg();
  file.seekg(0, file.beg);

  container.resize(nchars);

  if (!file.read(&container[0], nchars))
    throw std::runtime_error("cannot read file");

  file.close();
}

const std::array<uint8_t, 85> Lut = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04};

void pptable(const std::vector<uint32_t> &table) {
  printf("%4s %3s %3s %3s %3s\n", "A", "C", "G", "T", "X");
  uint32_t bound = std::min(table.size() / 5, 50UL);

  for (uint32_t ncols = 0; ncols < bound; ++ncols) {
    for (uint8_t idx = 0; idx < 5; ++idx)
      printf("%4d", table[5 * ncols + idx]);

    printf("\n");
  }
}

__global__ void match(uint32_t *d_table, char *d_source, uint32_t size,
                      uint8_t *d_lut, uint8_t *d_output) {
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
        d_output[wordidx - 1] = 0x31;

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

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: %s <file_with_genome_filepaths> <file_with_markers> "
            "<output_file>\n",
            argv[0]);
    return 1;
  }

  std::vector<std::string> sources;
  std::ifstream sourcesf(argv[1]);
  if (!sourcesf) {
    fprintf(stderr, "there is no %s to open\n", argv[1]);
    return 1;
  }

  std::string line;
  while (std::getline(sourcesf, line))
    sources.push_back(line);

  sourcesf.close();

  std::ifstream markersf(argv[2]);
  if (!markersf) {
    fprintf(stderr, "there is no %s to open\n", argv[2]);
    return 1;
  }

  std::vector<std::string> markers;
  uint64_t nchars = 0;
  while (std::getline(markersf, line)) {
    markers.emplace_back(line.begin() + line.find(',') + 1, line.end());

    nchars += line.size();
  }

  markersf.close();

  uint32_t tablesize = std::ceil(
      nchars -
      1 / 2 * markers.size() * std::log2(markers.size() / std::sqrt(4)) + 24);

  std::vector<uint32_t> table(tablesize * 5, 0);

  uint32_t edge = 0;
  uint32_t wordidx = 0;

  std::unordered_map<uint32_t, std::vector<uint32_t>> duplicates;
  std::unordered_map<std::string, uint32_t> marked_mapping;

  for (const auto &marker : markers) {
    uint32_t vx = 0;

    for (auto &base : marker) {
      uint32_t idx = 5 * vx + Lut[base] - 1;

      if (table[idx] == 0)
        table[idx] = ++edge;

      vx = table[idx];
    }

    auto search = marked_mapping.find(marker);

    ++wordidx;

    if (search == marked_mapping.end()) {
      table[5 * vx + 4] = wordidx;
      marked_mapping[marker] = wordidx;

      // trim this one, later
      duplicates[wordidx] = {};
    } else {
      duplicates[search->second].push_back(wordidx);
    }
  }

  std::ofstream outputf(argv[3]);
  std::string source;

  std::string sourcesfn(argv[1]);
  std::string prefix(sourcesfn.substr(0, sourcesfn.find_last_of('/') + 1));

  uint32_t *d_table;
  uint8_t *d_lut;
  char *d_source;
  uint8_t *d_output;

  cudaMalloc((void **)&d_table, table.size() * sizeof(uint32_t));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_lut, Lut.size());
  cudaMemcpy(d_lut, &Lut, Lut.size(), cudaMemcpyHostToDevice);

  // unfancy foreknowledge
  cudaMalloc((void **)&d_source, 150 * 1 << 20);
  std::vector<uint8_t> output(markers.size(), 0x30);
  cudaMalloc((void **)&d_output, output.size());

  for (std::string &sourcefn : sources) {
    {
      measure("reading");
      readfile(prefix + sourcefn, source);
    }

    {
      measure("working");
      cudaMemcpy(d_source, source.data(), source.size(),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(d_output, output.data(), output.size(),
                 cudaMemcpyHostToDevice);
      match<<<8000, 1024>>>(d_table, d_source, source.size(), d_lut, d_output);

      cudaMemcpy(output.data(), d_output, output.size(),
                 cudaMemcpyDeviceToHost);
    }

    {
      measure("processing");
      for (const auto &pair : duplicates) {
        if (output[pair.first - 1] == 0x31) {
          for (const auto &idx : pair.second)
            output[idx - 1] = 0x31;
        }
      }
    }

    {
      measure("writing");
      outputf << sourcefn.substr(sourcefn.find_last_of('/') + 1,
                                 sourcefn.size())
              << ' ';
      outputf.write((char *)output.data(), output.size());
      outputf << std::endl;

      std::fill(output.begin(), output.end(), 0x30);
    }
  }

  outputf.close();
  cudaFree(d_table);
  cudaFree(d_lut);
  cudaFree(d_output);
  cudaFree(d_source);
}
