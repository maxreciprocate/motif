#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jam.h"
#include "stdio.h"
#include "string.h"

#define debug 0

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

void pptable(const std::vector<uint32_t> &table) {
  printf("%4s %3s %3s %3s %3s\n", "A", "C", "G", "T", "X");
  uint32_t bound = std::min(table.size() / 5, 50UL);

  for (uint32_t ncols = 0; ncols < bound; ++ncols) {
    for (uint8_t idx = 0; idx < 5; ++idx)
      printf("%4d", table[5 * ncols + idx]);

    printf("\n");
  }
}

typedef std::pair<std::string, std::string> P;

class Queue {
private:
  std::vector<P> queue;
  std::condition_variable takecv;
  std::condition_variable addcv;
  std::mutex mx;
  std::atomic<int> count;
  size_t idx;
  bool done;

public:
  Queue(uint64_t size) : queue(size + 1), idx(0), done(false), count(0) {}
  inline void sub() {
    ++count;
  }

  void push(std::string sourcefn) {
    std::unique_lock<std::mutex> lock(mx);

    while (idx == queue.capacity() - 1)
      takecv.wait(lock);

    queue[++idx].first = sourcefn;
    readfile(sourcefn, queue[idx].second);

    lock.unlock();
    addcv.notify_one();
  }

  void push(P pair) {
    std::unique_lock<std::mutex> lock(mx);

    while (idx == queue.capacity() - 1)
      takecv.wait(lock);

    queue[++idx] = pair;

    lock.unlock();
    addcv.notify_one();
  }

  P pop() {
    std::unique_lock<std::mutex> lock(mx);

    while (!done && idx == 0)
      addcv.wait(lock);

    if (done && idx == 0)
      return std::make_pair("", "");

    auto element = queue[idx--];

    lock.unlock();
    takecv.notify_one();

    return element;
  }

  void finish() {
    if (--count > 0) return;

    done = true;
    addcv.notify_all();
  }
};

void process (Queue& sourcequeue, Queue& outputqueue, std::vector<uint32_t>& table, std::unordered_map<uint32_t, std::vector<uint32_t>>& duplicates, uint64_t markerssize, uint8_t deviceidx) {
  uint32_t *d_table;
  char *d_source;
  uint8_t *d_output;

  if (debug)
    printf("setting up %d device\n", deviceidx);

  auto error = cudaSetDevice(deviceidx);

  if (error != cudaSuccess) {
    printf("(cuda): can't select device #%d with %s\n", deviceidx, cudaGetErrorString(error));
    return;
  }

  setup(table);

  // unfancy foreknowledge
  cudaMalloc((void **)& d_source, 150 * 1 << 20);
  std::vector<uint8_t> output(markerssize, 0x30);
  cudaMalloc((void **)& d_output, output.size());

  outputqueue.sub();
  std::pair<std::string, std::string> pair = sourcequeue.pop();

  float timing = 0;
  float total  = 0;
  size_t times = 0;

  while (pair.second.size() > 0) {
    auto source = pair.second;
    auto sourcefn = pair.first;

    match(d_source, source, d_output, output, &timing);
    ++times;
    total += timing;

    for (const auto &pair : duplicates) {
      if (output[pair.first - 1] == 0x31) {
        for (const auto &idx : pair.second)
          output[idx - 1] = 0x31;
      }
    }

    outputqueue.push(make_pair(sourcefn, std::string(output.begin(), output.end())));

    std::fill(output.begin(), output.end(), 0x30);
    pair = sourcequeue.pop();
  }

  if (debug)
    printf("Average timing of match: %.2fms\n", total / times);

  outputqueue.finish();

  cudaFree(d_table);
  cudaFree(d_output);
  cudaFree(d_source);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: %s <file_with_genome_filepaths> <file_with_markers> <output_file>\n",
            argv[0]);
    return 1;
  }

  std::ifstream sourcesf(argv[1]);
  if (!sourcesf) {
    fprintf(stderr, "there is no %s to open\n", argv[1]);
    return 1;
  }

  std::vector<std::string> sourcesfns;
  std::string line;
  while (std::getline(sourcesf, line))
    sourcesfns.push_back(line);

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
      uint32_t idx = 5 * vx + Lut[base - 0x40] - 1;

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

  std::string sourcesfn(argv[1]);
  std::string prefix(sourcesfn.substr(0, sourcesfn.find_last_of('/') + 1));

  Queue sourcequeue (64);
  Queue outputqueue (64);

  std::thread reader {[&]() {
    for (auto& sourcesfn: sourcesfns)
      sourcequeue.push(prefix + sourcesfn);

    sourcequeue.finish();
  }};

  std::string outputfn (argv[3]);
  std::thread writer {[&]() {
    std::ofstream outputf(outputfn);

    auto outputpair = outputqueue.pop();

    while (outputpair.second.size() > 0) {
      auto sourcefn = outputpair.first;
      auto output = outputpair.second;

      outputf << sourcefn.substr(sourcefn.find_last_of('/') + 1, sourcefn.size())
              << ' ';
      outputf.write((char *) output.data(), output.size());
      outputf << std::endl;

      outputpair = outputqueue.pop();
    }

    outputf.close();
  }};

  int devicecount = 0;
  cudaError_t error = cudaGetDeviceCount(&devicecount);
  if (error != cudaSuccess) {
    printf("(cuda): can't get a grip upon devices with %s\n", cudaGetErrorString(error));
    exit(1);
  }

  std::vector<std::thread> workers;
  workers.reserve(devicecount);

  uint8_t offset = argc < 5 ? 0 : std::min(std::stoi(argv[4]), devicecount - 1);

  for (uint8_t idx = offset; idx < devicecount; ++idx) {
    workers.emplace_back(
      process,
      std::ref(sourcequeue), std::ref(outputqueue), std::ref(table),
      std::ref(duplicates), markers.size(), idx);
  }

  for (auto& t: workers)
    t.join();

  writer.join();
  reader.join();
}
