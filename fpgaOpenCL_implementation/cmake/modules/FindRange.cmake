
include(ExternalProject)
find_package(Git REQUIRED)

set(range_prefix "${CMAKE_BINARY_DIR}/_3rdParty/range")

ExternalProject_Add(
    range
    PREFIX ${range_prefix}
    GIT_REPOSITORY https://github.com/harrism/cpp11-range.git
    TIMEOUT 10
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_COMMAND ""
    LOG_DOWNLOAD ON
)

# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(range source_dir)
set(RANGE_INCLUDE_DIR ${source_dir} CACHE INTERNAL "Path to include folder for Range")
