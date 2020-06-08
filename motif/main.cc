#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h>
#include "jam/motif.h"
#include <stdio.h>

PYBIND11_MODULE(motif, m) {
  pybind11::class_<Motif>(m, "Motif")
    .def(pybind11::init<>())
    .def("build", &Motif::build)
    .def("run",   &Motif::run)
    .def("clear", &Motif::clear);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
