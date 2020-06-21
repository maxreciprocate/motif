#include "motif.h"
#include "queue.h"

std::array<uint8_t, 'T' - 'A' + 1> translate{{
  0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3
}};

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

void Motif::process(Queue<std::pair<size_t, encodedGenomeData>>& sourcequeue, uint64_t max_genome_length,
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

  noteError(cudaMalloc((void **)& d_source, max_genome_length + 32 * 8));
  noteError(cudaMalloc((void **)& d_output, output_matrix.shape(1)));

  auto pair = sourcequeue.pop();
  while (pair.second.data.size() > 0) {
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

void skipN(const std::string& genome, size_t& pos) {
  if (genome[pos] != 'N') return;

  for (pos++; pos < genome.size(); ++pos) {
    if (genome[pos] != 'N') {
        --pos;
        return;
    }
  }

  --pos;
}

char encode_3_chars(const std::string& chars) {
  uint8_t countN = 0;
  if (chars[0] == 'N') countN++;
  if (chars[1] == 'N') countN++;
  if (chars[2] == 'N') countN++;

  char encoded_char = 0;
  // data (chars) will be saved in "reverse" mode for easier future access by shifts and bitwise
  if (countN == 0) {
    // encode all 3 chars by 2 bits
    encoded_char |= translate[chars[2] - 'A'];
    encoded_char <<= 2;

    encoded_char |= translate[chars[1] - 'A'];
    encoded_char <<= 2;

    encoded_char |= translate[chars[0] - 'A'];
    encoded_char <<= 2;
  } else if (countN == 1) {
    // in this case we also need to save index of N
    uint8_t index = 0;
    if (chars[1] == 'N') index = 1;
    if (chars[2] == 'N') index = 2;

    // encode 2 "not N" chars
    for (int8_t i = 2; i >= 0; --i) {
        if (i == index) continue;
        encoded_char |= translate[chars[i] - 'A'];
        encoded_char <<= 2;
    }
    encoded_char |= index;
    encoded_char <<= 2;
  } else if (countN == 2) {
    // in these case ("N_N", since we replace "NN" by "N") encode only middle char
    encoded_char |= translate[chars[1] - 'A'];
    encoded_char <<= 2;
  } else { // countN == 3
    // will be created in future version of the algorithm, when we will analyze 'N'
  }

  // save amount of 'N' in encoded char
  encoded_char |= countN;

  return encoded_char;
}


encodedGenomeData Motif::read_genome_from_string(pybind11::handle source) {
  std::string d_source = PyUnicode_AsUTF8(source.ptr());

  encodedGenomeData data = {
    .data = std::string{},
    .real_size = 0
  };

  auto& genome = data.data;
  genome.reserve(d_source.size());

  size_t i = 0;
  size_t cursor = 0;

  std::string substr;
  substr.reserve(3);

  while (i < d_source.size()) {
    substr.clear();

    for (uint8_t c = 0; c < 3 && i < d_source.size(); ++c, ++i) {
        skipN(d_source, i);
        substr.push_back(d_source[i]);
    }

    if (substr.size() != 3) {
      // if genome size not divisible by 3 than just add 'A' (zeros in encoded format) to the end
      data.real_size = (genome.size() * 3) + substr.size();
      for (uint8_t j = 0; j < (3 - substr.size()); ++j) {
        substr.push_back('A');
      }
    } else {
      data.real_size = (genome.size() + 1) * 3;
    }

    char encoded_char = encode_3_chars(substr);
    genome.push_back(encoded_char);
  }


  genome.shrink_to_fit();

  return data;
}


std::string Motif::read_genome_from_numpy(pybind11::handle source) {
  const char chars[4] = {'A', 'C', 'G', 'T'};
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

  Queue<std::pair<size_t, encodedGenomeData>> sourcequeue (MAX_SOURCE_QUEUE_SIZE);

  std::vector<std::thread> readers;
  const uint8_t reader_num = 16;
  readers.reserve(reader_num);

  for (uint8_t tidx = 0; tidx < reader_num; ++tidx) {
    readers.emplace_back([&, tidx]{
      printf("tidx: %d\n", tidx);
      for (size_t i = tidx; i < genome_data.size(); i += reader_num) {
        encodedGenomeData buff;

        if (is_numpy) {
          //buff = read_genome_from_numpy(genome_data[i]);
        } else
          buff = read_genome_from_string(genome_data[i]);
          //if (buff.size() == 0) {
          //  std::cerr << "Bad Genome" << std::endl;
          //  return;
          //}

        sourcequeue.push(std::make_pair(i, buff));
      }
    });
  }

  std::vector<std::thread> workers;
  workers.reserve(this->gpu_counter);

  for (size_t deviceidx = 0; deviceidx < this->gpu_counter; ++deviceidx) {
    if (this->gpu_selected[deviceidx]) {
      workers.emplace_back(
        &Motif::process, this, std::ref(sourcequeue), max_genome_length, output_matrix, deviceidx
      );
    }
  }

  for (auto& t: readers)
    t.join();
  sourcequeue.finish();

  for (auto& t: workers)
    t.join();
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
