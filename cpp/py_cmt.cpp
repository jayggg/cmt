#include "rk.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>

namespace py = pybind11;


PYBIND11_MODULE(cmt, m) {

  py::module::import("ngsolve");

  m.attr("__name__") = "cmt";

  py::class_<RKC>(m, "RKC", "Explicit Runge-Kutta methods for Y' = F(t, Y) "
		  "with complex-valued Y(t) and F(t, Y)", py::dynamic_attr())

    .def(py::init<function<void(double, FlatVector<Complex>)>, int, int>())

    .def("F", &RKC::F)

    .def("SolveIVP", &RKC::SolveIVP)
    
    .def("SolveFlow", &RKC::SolveFlow)
    ;

  m.def("RunRKTests", &RunRKTests);
    
}

  
