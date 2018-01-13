cmake_minimum_required (VERSION 2.6)
project (MachineLearning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#Inlcude headers into the project
include_directories(include)

#add sources
file(GLOB SOURCES "src/*.cpp")

add_executable(MachineLearning ${SOURCES})
