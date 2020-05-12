#include <pybind11/pybind11.h>
#include "motif/motif.h"
#include "motif/src/readers/file_readers.h"
#include <sstream>
// #include "jam/jam.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h> 
// #include <cuda_runtime.h>

namespace py = pybind11;

void read_markers_wraper(MARKERS_DATA* markers_data, const std::string& filename) {
    std::ifstream f_markers(filename);

    read_markers(markers_data, f_markers);

    f_markers.close();

}

std::string AUTOMATON::toString() {
    std::stringstream ss;
    ss << "Automatons:\t" << std::endl;
        
    for (auto& i : this->automaton) ss << i << std::endl;
        
    ss << "Output links:\t" << std::endl;

    for (auto& i : this->output_links) 
        for (auto& j : i) ss << j << std::endl;
        ss << std::endl;
    
    return ss.str();
}

int AUTOMATON::size() {
  return this->automaton.size();
}

AUTOMATON create_automaton_dummy(const MARKERS_DATA& markersData) {
  AUTOMATON automaton{};
  for(auto& i: markersData.markers){
    automaton.automaton.push_back(rand());
    for(int i =0; i<1000000;++i){
      continue;
    };
  }

  return automaton;
}

const py::array_t<int8_t>& match_dummy(
  const std::string &source,
  const AUTOMATON& automaton,
  py::array_t<int8_t>& output
) {

  for (int i=0; i < automaton.automaton.size(); ++i) {
    *output.mutable_data(i) = 1;
  }

  return output;
}




PYBIND11_MODULE(groove, m) {
    py::class_<AUTOMATON>(m, "AUTOMATON")
    .def(py::init<>())
    .def_readwrite("automaton", &AUTOMATON::automaton)
    .def_readwrite("output_links", &AUTOMATON::output_links)   
    .def("__str__", &AUTOMATON::toString)
    .def("__len__", &AUTOMATON::size)
    ;

    py::class_<MARKERS_DATA>(m, "MARKERS_DATA")
    .def(py::init<>())
    .def_readwrite("sum_of_all_chars", &MARKERS_DATA::sum_of_all_chars, py::return_value_policy::reference)
    .def_readwrite("longest_marker_len", &MARKERS_DATA::longest_marker_len)
    .def_readwrite("markers", &MARKERS_DATA::markers)
    ;

    m.def("create_automaton", &create_automaton , py::return_value_policy::reference);

    m.def("read_markers", &read_markers_wraper, py::return_value_policy::reference);

    m.def("match", &match_dummy, py::return_value_policy::reference);

    // m.def("print_automaton", &print_automaton, py::return_value_policy::reference);

    // m.def("read_create", &read_create);

    // m.def("search", search_markrs);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
