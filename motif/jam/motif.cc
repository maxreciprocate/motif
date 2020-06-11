#include "motif.h"
#include "queue.h"

void Motif::build(const pybind11::list markers_data, const pybind11::array_t<int> gpu_devices) {
  if (markers_data.size() == 0) {
    std::cerr << "Empty list of markers" << std::endl;
    return;
  }
  std::vector<std::string> markers;
  markers.reserve(markers_data.size());

  uint64_t nchars = 0;

  for (ssize_t i = 0; i < markers_data.size(); ++i) {
    auto data = PyUnicode_AsUTF8(markers_data[i].ptr());

    if (!data) {
      std::cerr << "(build table): can't read marker number " << i << std::endl;
    }

    markers.emplace_back(data);

    nchars += strlen(data);
  }

  size_t tablesize = 5 * std::ceil(
      nchars -
      1 / 2 * markers.size() * std::log2(markers.size() / std::sqrt(4)) + 24);

  std::vector<uint32_t> table(tablesize, 0);

  uint32_t edge = 0;
  uint32_t wordidx = 0;

  auto& duplicates = this->duplicates;
  auto& marked_mapping = this->marked_mapping;

  for (const auto& marker: markers) {
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

      duplicates[wordidx] = {};
    } else {
      duplicates[search->second].push_back(wordidx);
    }
  }

  int devicecount = 0;

  cudaError_t error = cudaGetDeviceCount(&devicecount);
  if (error != cudaSuccess) {
    printf("(cuda): can't get a grip upon devices with %s\n", cudaGetErrorString(error));
    exit(1);
  }

  auto devices = gpu_devices.data();

  this->gpu_selected.assign(gpu_devices.data(), gpu_devices.data() + gpu_devices.shape(0));
  this->table_pointers.reserve(gpu_devices.shape(0));

  for (size_t deviceidx = 0; deviceidx < devicecount; ++deviceidx) {
    if (devices[deviceidx]) {
       auto& d_table = this->table_pointers[deviceidx];

       if (debug)
         printf("setting up %zu device\n", deviceidx);

       auto error = cudaSetDevice(deviceidx);

       if (error != cudaSuccess) {
         printf("(cuda): can't select device #%ld with %s\n", deviceidx, cudaGetErrorString(error));
         return;
       }

       setup(d_table, table);
       this->gpu_counter++;
    }
  }
  built = true;
}

void Motif::process(Queue<std::pair<int, std::string>>& sourcequeue, uint64_t max_genome_length,
                    pybind11::array_t<int8_t> output_matrix, size_t deviceidx) {
  if (debug)
    printf("setting up %zu device\n", deviceidx);

  auto error = cudaSetDevice(deviceidx);

  if (error != cudaSuccess) {
    fprintf(stderr, "(cuda): can't select device #%ld with %s\n", deviceidx, cudaGetErrorString(error));
    return;
  }

  char* d_source;
  int8_t* d_output;

  noteError(cudaMalloc((void **)& d_source, max_genome_length));
  noteError(cudaMalloc((void **)& d_output, output_matrix.shape(1)));

  auto pair = sourcequeue.pop();
  while (pair.second.size() > 0) {
    auto source = pair.second;
    auto source_idx = pair.first;
    auto output_row = output_matrix.mutable_data(source_idx);

    match(d_source, source, d_output, output_row, output_matrix.shape(1));

    for (const auto &pair : this->duplicates) {
      if (output_row[pair.first - 1] == 1) {
        for (const auto &idx : pair.second)
          output_row[idx - 1] = 1;
      }
    }

    pair = sourcequeue.pop();
  }

  noteError(cudaFree(d_output));
  noteError(cudaFree(d_source));
}

std::string Motif::read_genome_from_string(pybind11::handle source) {
  std::string genome = PyUnicode_AsUTF8(source.ptr());

  size_t j = 0;

  for (int i = 0; i < genome.size() - 1; ++i) {
    auto ch = genome[i];

    if (ch == 'N' && genome[i + 1] == 'N') continue;

    genome[j++] = ch;
  }

  genome[j++] = genome[genome.size() - 1];

  genome.resize(j);
  return genome;
}


std::string Motif::read_genome_from_numpy(pybind11::handle source) {
  const char chars[4] = {'A', 'T', 'C', 'G'};
  auto data = pybind11::array_t<int8_t, pybind11::array::c_style | pybind11::array::forcecast>::ensure(source);
  if (data.shape(1) < 1) {
    std::cerr << "Empty Genome" << std::endl;
    return "";
  }
  std::string buff;

  buff.reserve(data.shape(1));

  for (size_t col = 0; col < data.shape(1); ++col) {
    char ch = 'N';

    for (size_t row = 0; row < data.shape(0); ++row) {
      if (*data.data(row, col) == 1) {
        ch = chars[row];
        break;
      }
    }

    buff.push_back(ch);
  }

  return buff;
}

void Motif::run(const pybind11::list genome_data, uint64_t max_genome_length,
                pybind11::array_t<int8_t> output_matrix, bool is_numpy) {

  if (!this->built) {
    std::cerr << "call Motif.build() befor calling run" << std::endl;
    return;
  }

  Queue<std::pair<int, std::string>> sourcequeue (MAX_SOURCE_QUEUE_SIZE);

  std::thread reader {[&]() {
    for (size_t i = 0; i < genome_data.size(); ++i) {
      std::string buff;

      if (is_numpy)
        buff = read_genome_from_numpy(genome_data[i]);
      else
        buff = read_genome_from_string(genome_data[i]);
        if (buff.size() == 0) {
          std::cerr << "Bad Genome" << std::endl;
          return;
        }

      sourcequeue.push(std::make_pair(i, buff));
    }

    sourcequeue.finish();
  }};

  std::vector<std::thread> workers;
  workers.reserve(this->gpu_counter);

  for (size_t deviceidx = 0; deviceidx < this->gpu_counter; ++deviceidx) {
    if (this->gpu_selected[deviceidx]) {
      workers.emplace_back(
        &Motif::process, this, std::ref(sourcequeue), max_genome_length, output_matrix, deviceidx
      );
    }
  }

  for (auto& t: workers)
    t.join();

  reader.join();
}

void Motif::clear() {
  if (this->cleared) return;

  for (size_t deviceidx = 0; deviceidx < this->gpu_counter; ++deviceidx) {
    if (this->gpu_selected[deviceidx]) {

      if (debug)
        printf("setting up %zu device\n", deviceidx);

      auto error = cudaSetDevice(deviceidx);

      if (error != cudaSuccess) {
        printf("(cuda): can't select device #%ld with %s\n", deviceidx, cudaGetErrorString(error));
        return;
      }

      clear_table(this->table_pointers[deviceidx]);
    }
  }

  this->cleared = true;
};
