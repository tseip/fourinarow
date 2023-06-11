cmake_minimum_required(VERSION 3.15)
project(NInARow)

include (FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Make the default build type Release. If user or another
# project sets a different value than use that
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to default -- Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()
message(STATUS "NInARow Build Type: ${CMAKE_BUILD_TYPE}")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(SWIG REQUIRED)
find_package(PythonLibs REQUIRED)

include(${SWIG_USE_FILE})
set(CMAKE_SWIG_FLAGS "")
include_directories(${PYTHON_INCLUDE_DIRS})

set(CXX_FLAGS)

list(APPEND CXX_FLAGS "-fPIC" "-Wall" "-Wextra" "-Werror" "-Wno-unused" "-fexceptions")

enable_testing()

add_executable(tests
  bfs_node_ut.cpp
  ninarow_bfs_ut.cpp
  ninarow_board_ut.cpp
  ninarow_heuristic_ut.cpp
  ninarow_heuristic_feature_ut.cpp
  ninarow_move_ut.cpp
  ninarow_pattern_ut.cpp
)
target_link_libraries(tests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)

SET_SOURCE_FILES_PROPERTIES(fourbynine.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_LIBRARY(swig_fourbynine TYPE SHARED LANGUAGE python OUTPUT_DIR "../model_fitting" OUTFILE_DIR "./" SOURCES fourbynine.i)
set_property(TARGET swig_fourbynine PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
target_include_directories(swig_fourbynine PUBLIC "./")
target_link_libraries(swig_fourbynine ${PYTHON_LIBRARIES})

add_custom_command(TARGET swig_fourbynine POST_BUILD 
  COMMAND "${CMAKE_COMMAND}" -E copy 
     "$<TARGET_FILE:swig_fourbynine>"
     "../model_fitting" 
  COMMENT "Copying SWIG library to output directory")
