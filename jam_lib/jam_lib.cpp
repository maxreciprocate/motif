#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h>

void run(
  const pybind11::list genome_data,
  const uint64_t max_genome_length,
  const pybind11::list markers_data,
  pybind11::array_t<int8_t> output_matrix,
  const pybind11::array_t<int> gpu_devices,
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
