#include <pybind11/pybind11.h>
#include <sstream>
// #include "jam/jam.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h> 
#include <cuda_runtime.h>

namespace py = pybind11;
void run(const std::string& genome_path,
        const std::string& markers_path,
        const std::string& out_path,
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
