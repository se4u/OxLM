# CMake Configuration File for OxLM Project.
cmake_minimum_required(VERSION 2.6)
project(OxLM)

#############################################
# Directories
#############################################

# Store all binaries in bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)

# Load additional CMake modules from cmake/ directory
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

# Include project path (src/)
include_directories(${PROJECT_SOURCE_DIR})

#############################################
# Libraries
#############################################

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS serialization program_options
  date_time REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
# link_directories(${Boost_LIBRARY_DIRS})
# set(link_libraries(train_cnlm ${Boost_LIBRARIES} cnlm)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP QUIET)

#############################################
# Hackery courtesy of Karl Moritz
#############################################

# Overrides standard add_executable command and by default links all executables
# to boost. This will need changing only should we end up including code not
# reliant on boost. As if that was ever going to happen ...
macro (add_executable _name)
  _add_executable(${ARGV})
  if (TARGET ${_name})
    target_link_libraries(${_name} ${Boost_LIBRARIES})
  endif()
endmacro()

#############################################
# Compiler Settings
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_subdirectory(cg)
add_subdirectory(experimental/kmh)
add_subdirectory(experimental/egrefen)