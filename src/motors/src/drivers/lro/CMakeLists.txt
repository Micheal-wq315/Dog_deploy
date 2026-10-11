AUX_SOURCE_DIRECTORY(. SOURCE_LIST_LRO_MOTORS)
add_library(lro_motors
        STATIC
        ${SOURCE_LIST_LRO_MOTORS}
)
target_include_directories(lro_motors PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(lro_motors PUBLIC ${PUBLIC_DEPENDENCIES} motors_can motors_canfd)

option(BUILD_LRO_TEST "Build LRO motor test programs" OFF)
if(BUILD_LRO_TEST)
    add_subdirectory(test)
endif()
