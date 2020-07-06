project(_base_bindings CXX)

set(SRC ${CMAKE_CURRENT_LIST_DIR}/base_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${common_includes})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
