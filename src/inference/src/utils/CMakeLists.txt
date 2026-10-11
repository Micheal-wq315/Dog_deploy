# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE ./ ${CNPY_INCLUDE_DIR})
target_link_libraries(utils INTERFACE ${PUBLIC_DEPENDENCIES})
