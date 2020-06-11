#ifndef MOTIF_H
#define MOTIF_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include "jam.h"
#include "queue.h"

class Motif {
private:
  std::unordered_map<uint32_t, std::vector<uint32_t>> duplicates;
  std::unordered_map<std::string, uint32_t> marked_mapping;
  std::vector<uint32_t*> table_pointers;
  std::vector<bool> gpu_selected;
  size_t gpu_counter;
  bool cleared;
  bool built;

public:
  Motif() = default;
  Motif(const Motif&) = delete;

  void build(
    const pybind11::list markers_data,
    const pybind11::array_t<int> gpu_devices
  );

  void run(
    const pybind11::list genome_data,
    uint64_t max_genome_length,
    pybind11::array_t<int8_t> output_matrix,
    bool is_numpy
  );

  void process(
    Queue<std::pair<int, std::string>>& sourcequeue,
    uint64_t max_genome_length,
    pybind11::array_t<int8_t> output_matrix,
    size_t deviceidx
  );

  std::string read_genome_from_string(
    pybind11::handle source
  );

  std::string read_genome_from_numpy(
    pybind11::handle source
  );

  void clear();
  ~Motif() { clear(); }
};

#endif
