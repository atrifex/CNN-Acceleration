if(DEFINED SRC_CMAKE_)
  return()
else()
  set(SRC_CMAKE_ 1)
endif()

set(
  SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/main.cu
  ${CMAKE_CURRENT_LIST_DIR}/utils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/range.hpp
)
