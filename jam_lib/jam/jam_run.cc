// #include "jam_run.h"
#include "stdio.h"
#include "string.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream> 
#include <iostream>
#include <string>
// #include <string_view>
#include <unordered_map>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include "jam.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mutex>

namespace py = pybind11;


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

template <class T>
class Queue {
private:
  std::vector<T> queue;
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

  void push(T pair) {
    std::unique_lock<std::mutex> lock(mx);

    while (idx == queue.capacity() - 1)
      takecv.wait(lock);

    queue[++idx] = pair;

    lock.unlock();
    addcv.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mx);

    while (!done && idx == 0)
      addcv.wait(lock);

    if (done && idx == 0) {
      return T();
    }
      // return std::make_pair("", std::vector<int8_t>());

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


void process (Queue<std::pair<int, std::string>>& sourcequeue,
              py::array_t<int8_t> output_matrix,
              std::vector<uint32_t>& table, 
              std::unordered_map<uint32_t, std::vector<uint32_t>>& duplicates,
              uint64_t markerssize, uint8_t deviceidx) 
  {
  uint32_t *d_table;
  uint8_t *d_lut;
  char *d_source;
  int8_t *d_output;

  printf("setting up %d device\n", deviceidx);
  auto error = cudaSetDevice(deviceidx);

  if (error != cudaSuccess) {
    printf("(cuda): can't select device #%d with %s\n", deviceidx, cudaGetErrorString(error));
    return;
  }
  
  setup(table);
  // unfancy foreknowledge
  cudaMalloc((void **)& d_source, 150 * 1 << 20);
  cudaMalloc((void **)& d_output, output_matrix.shape(1));

  auto pair = sourcequeue.pop();
  float timing = 0;
  while (pair.second.size() > 0) {

    auto source = pair.second;
    auto source_idx = pair.first;
    auto output_row = output_matrix.mutable_data(source_idx);

    match(d_source, source, d_output, 
          output_row, output_matrix.shape(1), &timing);
    
    for (const auto &pair : duplicates) {
      if (output_row[pair.first - 1] == 1) {
        for (const auto &idx : pair.second)
          output_row[idx - 1] = 1;
      }
    }

    pair = sourcequeue.pop();
  }

  cudaFree(d_table);  
  cudaFree(d_lut);
  cudaFree(d_output);
  cudaFree(d_source);
}

void read_genome_from_numpy(py::handle source, std::string& buff) {
  static const char chars[4] = {'A', 'T', 'C', 'G'};
  auto data = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(source);

  buff.reserve(data.shape(1));

  for (int i = 0; i < data.shape(1); ++i) {

    char ch = 'N';
    for (int j = 0; j < data.shape(0); ++j) {
      if (*data.data(j, i) == 1) {
        ch = chars[j];
      }
    }
    buff.push_back(ch); 
  }
}


void run(
  const py::list genome_data,
  const py::list markers_data,
  py::array_t<int8_t> output_matrix,
  int n_devices,
  bool is_numpy
)
{
  std::vector<std::string> markers;
  uint64_t nchars = 0;

  for (ssize_t i = 0; i < markers_data.size(); ++i) {

    auto data = PyUnicode_AsUTF8(markers_data[i].ptr());

    if(!data) {
      std::cerr << "failed marker" << std::endl;
    }

    markers.emplace_back(data);

    nchars += strlen(data);
  }

  uint32_t tablesize = std::ceil(
      nchars -
      1 / 2 * markers.size() * std::log2(markers.size() / std::sqrt(4)) + 24);

  std::vector<uint32_t> table(tablesize * 5, 0);

  uint32_t edge = 0;
  uint32_t wordidx = 0;

  std::unordered_map<uint32_t, std::vector<uint32_t>> duplicates;
  std::unordered_map<std::string, uint32_t> marked_mapping;

  for (const auto& marker: markers) {

    uint32_t vx = 0;

    for (auto &base : marker) {
      uint32_t idx = 5 * vx + Lut[base- 0x40] - 1;

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

  Queue<std::pair<int, std::string>> sourcequeue (64);

  std::thread reader {[&]() {
    for (int i = 0; i < genome_data.size(); ++i) {
      std::string buff;
      if (is_numpy) 
        read_genome_from_numpy(genome_data[i], buff); 
      else 
        buff = PyUnicode_AsUTF8(genome_data[i].ptr());
      sourcequeue.push(std::make_pair(i, buff));
    }
    sourcequeue.finish();
  }};

  int devicecount = 0;
  cudaError_t error = cudaGetDeviceCount(&devicecount);
  if (error != cudaSuccess) {
    printf("(cuda): can't get a grip upon devices with %s\n", cudaGetErrorString(error));
    exit(1);
  }

  std::vector<std::thread> workers;
  workers.reserve(devicecount);
  uint8_t offset = std::min(n_devices, devicecount-1);
  for (uint8_t idx = offset; idx < devicecount; ++idx) {
    workers.emplace_back(process, std::ref(sourcequeue), output_matrix, std::ref(table), std::ref(duplicates), markers.size(), idx);
  }

  for (auto& t: workers)
    t.join();

  reader.join();
}
