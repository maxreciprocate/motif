#include <pybind11/pybind11.h>
#include <sstream>
// #include "jam/jam.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h> 
#include <cuda_runtime.h>

namespace py = pybind11;


void run(
  // const py::list genome_name,
  const py::list genome_data,
  const py::list markers_data,
  py::array_t<int8_t> output_matrix,
  // py::list output_matrix,
  int n_devices
);

PYBIND11_MODULE(jam_lib, m) {
  m.def("run", &run);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
