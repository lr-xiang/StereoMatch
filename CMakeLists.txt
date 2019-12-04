# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

# Project's name
project(stereo CXX)
set(CMAKE_BUILD_TYPE "RELEASE")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3" )
# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -O3")

#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} "-O2")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} "-O2")

set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
#MESSAGE(STATUS "OPENCV_INCLUDE: " ${OpenCV_INCLUDE_DIRS})
#MESSAGE(STATUS "OPENCV_LIBS: " ${OpenCV_LIBS})
set(PCL_DIR "/usr/local/share/pcl-1.8")
find_package(PCL REQUIRED NO_DEFAULT_PATH)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Source files
#file(GLOB SRC ./src/*.cpp)
#FILE(GLOB_RECURSE HEADERS "${PROJECT_SOURCE_DIR}/include/*.hpp")
add_custom_target(headers SOURCES ${HEADERS})

#MESSAGE(STATUS "SRC_FILES: " ${SOURCES})

# The following folder will be included
include(FindCUDA)
include_directories(/usr/local/cuda/include)
include_directories("${PROJECT_SOURCE_DIR}/include" ${PCL_INCLUDE_DIRS})

#add_executable(CameraCalibration camera_calibration.cpp)
#
#target_link_libraries(CameraCalibration ${OpenCV_LIBS})

cuda_add_executable(StereoTest src/stereo_opencv.cpp StereoMatching.h StereoMatching.cpp pm.h pm.cpp)
target_link_libraries(StereoTest ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcurand.so)

#cuda_add_executable(StereoTest src/stereo_opencv.cpp StereoMatching.h StereoMatching.cpp pm.h pm.cpp PatchMatchStereoGPU.cu lodepng.cpp)

#PatchMatchStereoGPU.h PatchMatchStereoGPU.cu lodepng.h lodepng.cpp
#target_link_libraries(StereoTest ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})


#/usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcurand.so




