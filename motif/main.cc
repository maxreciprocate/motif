#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h>
#include "jam/motif.h"
#include <stdio.h>


PYBIND11_MODULE(motif, m) {
  pybind11::class_<Motif>(m, "Motif")
    .def(pybind11::init<>())
    .def("build", &Motif::build,
    "A function which builds all necessary objects for making analysis. Needs to be called before calling 'run'",
    pybind11::arg("markers"), pybind11::arg("active_devices"))
    .def("run",   &Motif::run,
    "A function to perform the analysis",
    pybind11::arg("genomes"), pybind11::arg("max_genome_length"),
    pybind11::arg("output_matrix"), pybind11::arg("is_numpy"))
    .def("clear", &Motif::clear, "A function to clear all data from ");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
