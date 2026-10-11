cmake_minimum_required(VERSION 3.12)
project(roboparty_motors)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake QUIET)
find_package(Boost COMPONENTS system)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# pybind11_add_module 默认链接 pybind11::lto（注入 -flto 串行 LTO），
# 会让 lto-wrapper 报 "serial compilation of N LTRANS jobs" 且拖慢构建。
# motors_py 只是对已 -O3 的 motors 静态库做薄封装，LTO 无收益，直接关闭。
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
find_package(pybind11 REQUIRED)

set(PUBLIC_DEPENDENCIES
    fmt::fmt ${Boost_LIBRARIES} pthread Eigen3::Eigen)

add_subdirectory(src)

add_library(motors STATIC 
  src/motor_driver.cpp
)

target_include_directories(motors
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(motors PUBLIC ${PUBLIC_DEPENDENCIES}
  lro_motors rs02_motors
  motors_can motors_canfd
)

install(DIRECTORY include/ DESTINATION include)

install(TARGETS motors lro_motors rs02_motors motors_can motors_canfd
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install our custom Config for non-ROS consumers (Deb mode)
install(FILES cmake/roboparty_motorsConfig.cmake
  DESTINATION lib/cmake/roboparty_motors
)

pybind11_add_module(motors_py
  src/pybind_module.cpp
)

target_include_directories(motors_py
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(motors_py PUBLIC motors)

set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(TARGETS motors_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  ARCHIVE DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${PYTHON_INSTALL_DIR}
)

if(ament_cmake_FOUND)
    ament_export_libraries(motors lro_motors rs02_motors motors_can motors_canfd)
    ament_export_include_directories(include)
    ament_package()
endif()
