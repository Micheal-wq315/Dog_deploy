AUX_SOURCE_DIRECTORY(. SOURCE_LIST_RS02_MOTORS)
add_library(rs02_motors
        STATIC
        ${SOURCE_LIST_RS02_MOTORS}
)
target_include_directories(rs02_motors PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rs02_motors PUBLIC ${PUBLIC_DEPENDENCIES} motors_can)
