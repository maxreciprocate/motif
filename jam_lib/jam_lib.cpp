#include <pybind11/pybind11.h>
#include <sstream>
// #include "jam/jam.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h> 
#include <cuda_runtime.h>

namespace py = pybind11;

void run(
  const py::list genome_data,
  const uint64_t max_genome_length,
  const py::list markers_data,
  py::array_t<int8_t> output_matrix,
  int n_devices,
  bool is_numpy
);

PYBIND11_MODULE(jam_lib, m) {
  m.def("run", &run);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
