cmake_minimum_required(VERSION 3.26)
project(Project_Code)

set(CMAKE_CXX_STANDARD 17)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add executable and link OpenMP
add_executable(Project_Code main.cpp)
target_link_libraries(Project_Code PRIVATE OpenMP::OpenMP_CXX)
# Set the number of threads using an environment variable
set(ENV{OMP_NUM_THREADS} "8")  # Set the desired number of threads here