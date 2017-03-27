if(DEFINED SRC_CMAKE_)
  return()
else()
  set(SRC_CMAKE_ 1)
endif()

set(
  SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/utils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/range.hpp
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/main.cl ${CMAKE_CURRENT_BINARY_DIR}/main.cl COPYONLY)
