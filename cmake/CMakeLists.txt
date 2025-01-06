cmake_minimum_required(VERSION 3.18)

include(ngsolve_addon.cmake)

# module name must match the argument of PYBIND11_MODULE
set(addon_name cmt)

# source files
add_ngsolve_addon(${addon_name}
	../cpp/py_cmt.cpp
	../cpp/rk_tests.cpp
	)

# install
set(install_dir ${ADDON_INSTALL_DIR_PYTHON}/${addon_name})
install(TARGETS ${addon_name} DESTINATION ${install_dir})
install(FILES ../py/__init__.py DESTINATION ${install_dir})

