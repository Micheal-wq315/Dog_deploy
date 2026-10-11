# roboparty_motorsConfig.cmake
#
# Config file for roboparty_motors
#
# Defines the IMPORTED target: roboparty_motors::roboparty_motors

# Check if we already have the target
if(TARGET roboparty_motors::roboparty_motors)
    return()
endif()

# Compute the installation prefix relative to this file
get_filename_component(CURRENT_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_INSTALL_PREFIX "${CURRENT_DIR}/../../../" ABSOLUTE)

set(roboparty_motors_INCLUDE_DIR "${_INSTALL_PREFIX}/include")
set(roboparty_motors_LIB_DIR     "${_INSTALL_PREFIX}/lib")

# Find dependencies
include(CMakeFindDependencyMacro)
find_dependency(fmt)
# spdlog replaced by built-in logger.hpp (backed by fmt)
find_dependency(Eigen3)

# Helper function to find and add libraries
function(_add_imported_lib _lib_name)
    find_library(LIB_${_lib_name}
        NAMES ${_lib_name}
        PATHS ${roboparty_motors_LIB_DIR}
        NO_DEFAULT_PATH
    )
    if(LIB_${_lib_name})
        list(APPEND roboparty_motors_LIBRARIES ${LIB_${_lib_name}})
        set(roboparty_motors_LIBRARIES ${roboparty_motors_LIBRARIES} PARENT_SCOPE)
    else()
        message(WARNING "Could not find roboparty_motors library: ${_lib_name}")
    endif()
endfunction()

# Find all component libraries
set(roboparty_motors_LIBRARIES "")
_add_imported_lib(motors)
_add_imported_lib(lro_motors)
_add_imported_lib(rs02_motors)
_add_imported_lib(motors_can)
_add_imported_lib(motors_canfd)

# Create the INTERFACE target
add_library(roboparty_motors::roboparty_motors INTERFACE IMPORTED)

target_include_directories(roboparty_motors::roboparty_motors INTERFACE ${roboparty_motors_INCLUDE_DIR})
target_link_libraries(roboparty_motors::roboparty_motors INTERFACE
    ${roboparty_motors_LIBRARIES}
    fmt::fmt
    Eigen3::Eigen
)

message(STATUS "Found roboparty_motors: ${_INSTALL_PREFIX}")
