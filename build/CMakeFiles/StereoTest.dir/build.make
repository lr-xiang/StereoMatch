# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/media/lietang/phenobot data/FLIR_stereo"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/media/lietang/phenobot data/FLIR_stereo/build"

# Include any dependencies generated for this target.
include CMakeFiles/StereoTest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/StereoTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/StereoTest.dir/flags.make

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o: CMakeFiles/StereoTest.dir/flags.make
CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o: ../src/stereo_opencv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/lietang/phenobot data/FLIR_stereo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o -c "/media/lietang/phenobot data/FLIR_stereo/src/stereo_opencv.cpp"

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/lietang/phenobot data/FLIR_stereo/src/stereo_opencv.cpp" > CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.i

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/lietang/phenobot data/FLIR_stereo/src/stereo_opencv.cpp" -o CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.s

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.requires:

.PHONY : CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.requires

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.provides: CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoTest.dir/build.make CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.provides.build
.PHONY : CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.provides

CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.provides.build: CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o


CMakeFiles/StereoTest.dir/StereoMatching.cpp.o: CMakeFiles/StereoTest.dir/flags.make
CMakeFiles/StereoTest.dir/StereoMatching.cpp.o: ../StereoMatching.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/lietang/phenobot data/FLIR_stereo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/StereoTest.dir/StereoMatching.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoTest.dir/StereoMatching.cpp.o -c "/media/lietang/phenobot data/FLIR_stereo/StereoMatching.cpp"

CMakeFiles/StereoTest.dir/StereoMatching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoTest.dir/StereoMatching.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/lietang/phenobot data/FLIR_stereo/StereoMatching.cpp" > CMakeFiles/StereoTest.dir/StereoMatching.cpp.i

CMakeFiles/StereoTest.dir/StereoMatching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoTest.dir/StereoMatching.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/lietang/phenobot data/FLIR_stereo/StereoMatching.cpp" -o CMakeFiles/StereoTest.dir/StereoMatching.cpp.s

CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.requires:

.PHONY : CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.requires

CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.provides: CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoTest.dir/build.make CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.provides.build
.PHONY : CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.provides

CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.provides.build: CMakeFiles/StereoTest.dir/StereoMatching.cpp.o


CMakeFiles/StereoTest.dir/pm.cpp.o: CMakeFiles/StereoTest.dir/flags.make
CMakeFiles/StereoTest.dir/pm.cpp.o: ../pm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/lietang/phenobot data/FLIR_stereo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/StereoTest.dir/pm.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoTest.dir/pm.cpp.o -c "/media/lietang/phenobot data/FLIR_stereo/pm.cpp"

CMakeFiles/StereoTest.dir/pm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoTest.dir/pm.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/lietang/phenobot data/FLIR_stereo/pm.cpp" > CMakeFiles/StereoTest.dir/pm.cpp.i

CMakeFiles/StereoTest.dir/pm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoTest.dir/pm.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/lietang/phenobot data/FLIR_stereo/pm.cpp" -o CMakeFiles/StereoTest.dir/pm.cpp.s

CMakeFiles/StereoTest.dir/pm.cpp.o.requires:

.PHONY : CMakeFiles/StereoTest.dir/pm.cpp.o.requires

CMakeFiles/StereoTest.dir/pm.cpp.o.provides: CMakeFiles/StereoTest.dir/pm.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoTest.dir/build.make CMakeFiles/StereoTest.dir/pm.cpp.o.provides.build
.PHONY : CMakeFiles/StereoTest.dir/pm.cpp.o.provides

CMakeFiles/StereoTest.dir/pm.cpp.o.provides.build: CMakeFiles/StereoTest.dir/pm.cpp.o


# Object files for target StereoTest
StereoTest_OBJECTS = \
"CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o" \
"CMakeFiles/StereoTest.dir/StereoMatching.cpp.o" \
"CMakeFiles/StereoTest.dir/pm.cpp.o"

# External object files for target StereoTest
StereoTest_EXTERNAL_OBJECTS =

StereoTest: CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o
StereoTest: CMakeFiles/StereoTest.dir/StereoMatching.cpp.o
StereoTest: CMakeFiles/StereoTest.dir/pm.cpp.o
StereoTest: CMakeFiles/StereoTest.dir/build.make
StereoTest: /usr/local/cuda/lib64/libcudart_static.a
StereoTest: /usr/lib/x86_64-linux-gnu/librt.so
StereoTest: /usr/local/lib/libopencv_dnn.so.3.4.3
StereoTest: /usr/local/lib/libopencv_superres.so.3.4.3
StereoTest: /usr/local/lib/libopencv_viz.so.3.4.3
StereoTest: /usr/local/lib/libopencv_videostab.so.3.4.3
StereoTest: /usr/local/lib/libopencv_stitching.so.3.4.3
StereoTest: /usr/local/lib/libopencv_shape.so.3.4.3
StereoTest: /usr/local/lib/libopencv_photo.so.3.4.3
StereoTest: /usr/local/lib/libopencv_objdetect.so.3.4.3
StereoTest: /usr/local/lib/libopencv_ml.so.3.4.3
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_system.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_thread.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_regex.so
StereoTest: /usr/lib/x86_64-linux-gnu/libpthread.so
StereoTest: /usr/local/lib/libpcl_common.so
StereoTest: /usr/local/lib/libpcl_octree.so
StereoTest: /usr/lib/libOpenNI.so
StereoTest: /usr/lib/libOpenNI2.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoradec.so
StereoTest: /usr/lib/x86_64-linux-gnu/libogg.so
StereoTest: /usr/lib/x86_64-linux-gnu/libz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
StereoTest: /usr/lib/x86_64-linux-gnu/libexpat.so
StereoTest: /usr/lib/x86_64-linux-gnu/libjpeg.so
StereoTest: /usr/lib/x86_64-linux-gnu/libpng.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtiff.so
StereoTest: /usr/lib/x86_64-linux-gnu/libfreetype.so
StereoTest: /usr/lib/libvtkWrappingTools-6.2.a
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
StereoTest: /usr/lib/x86_64-linux-gnu/libsz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libdl.so
StereoTest: /usr/lib/x86_64-linux-gnu/libm.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
StereoTest: /usr/lib/openmpi/lib/libmpi.so
StereoTest: /usr/lib/libgl2ps.so
StereoTest: /usr/lib/x86_64-linux-gnu/libxml2.so
StereoTest: /usr/lib/x86_64-linux-gnu/libpython2.7.so
StereoTest: /usr/local/lib/libpcl_io.so
StereoTest: /usr/local/lib/libpcl_stereo.so
StereoTest: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
StereoTest: /usr/local/lib/libpcl_kdtree.so
StereoTest: /usr/local/lib/libpcl_search.so
StereoTest: /usr/local/lib/libpcl_sample_consensus.so
StereoTest: /usr/local/lib/libpcl_filters.so
StereoTest: /usr/local/lib/libpcl_features.so
StereoTest: /usr/local/lib/libpcl_registration.so
StereoTest: /usr/local/lib/libpcl_ml.so
StereoTest: /usr/local/lib/libpcl_segmentation.so
StereoTest: /usr/local/lib/libpcl_visualization.so
StereoTest: /usr/lib/x86_64-linux-gnu/libqhull.so
StereoTest: /usr/local/lib/libpcl_surface.so
StereoTest: /usr/local/lib/libpcl_keypoints.so
StereoTest: /usr/local/lib/libpcl_tracking.so
StereoTest: /usr/local/lib/libpcl_recognition.so
StereoTest: /usr/local/lib/libpcl_people.so
StereoTest: /usr/local/lib/libpcl_outofcore.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_system.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_thread.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
StereoTest: /usr/lib/x86_64-linux-gnu/libboost_regex.so
StereoTest: /usr/lib/x86_64-linux-gnu/libpthread.so
StereoTest: /usr/lib/x86_64-linux-gnu/libqhull.so
StereoTest: /usr/lib/libOpenNI.so
StereoTest: /usr/lib/libOpenNI2.so
StereoTest: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoradec.so
StereoTest: /usr/lib/x86_64-linux-gnu/libogg.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallelLIC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingLIC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
StereoTest: /usr/lib/x86_64-linux-gnu/libexpat.so
StereoTest: /usr/lib/x86_64-linux-gnu/libjpeg.so
StereoTest: /usr/lib/x86_64-linux-gnu/libpng.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtiff.so
StereoTest: /usr/lib/x86_64-linux-gnu/libfreetype.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeOpenGL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkWrappingJava-6.2.so.6.2.0
StereoTest: /usr/lib/libvtkWrappingTools-6.2.a
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOFFMPEG-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOMovie-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelStatistics-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingMath-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOMINC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
StereoTest: /usr/lib/x86_64-linux-gnu/libsz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libdl.so
StereoTest: /usr/lib/x86_64-linux-gnu/libm.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
StereoTest: /usr/lib/openmpi/lib/libmpi.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtSQL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkViewsGeovis-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingStatistics-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersSelection-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOVPIC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkVPIC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOAMR-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneric-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeOpenGL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOImport-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOParallel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIONetCDF-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.2.so.6.2.0
StereoTest: /usr/lib/libgl2ps.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOXdmf2-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkxdmf2-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libxml2.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtOpenGL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingExternal-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingImage-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelImaging-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkTestingRendering-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOVideo-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersPython-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libpython2.7.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelMPI-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOODBC-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOMySQL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOGeoJSON-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOParallelXML-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkTestingGenericBridge-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersReebGraph-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOParallelExodus-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelGeometry-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkLocalExample-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOEnSight-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOParallelNetCDF-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersSMP-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOPostgreSQL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersHyperTree-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOParallelLSDyna-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersVerdict-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOInfovis-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingStencil-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkTestingIOSQL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOGDAL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOMPIParallel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOMPIImage-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI4Py-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelFlowPaths-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkDomainsChemistry-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersProgrammable-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingMorphological-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtWebkit-6.2.so.6.2.0
StereoTest: /usr/local/cuda/lib64/libcudart_static.a
StereoTest: /usr/lib/x86_64-linux-gnu/librt.so
StereoTest: /usr/local/cuda/lib64/libcudart.so
StereoTest: /usr/local/cuda/lib64/libcurand.so
StereoTest: /usr/local/lib/libpcl_common.so
StereoTest: /usr/local/lib/libpcl_octree.so
StereoTest: /usr/local/lib/libpcl_io.so
StereoTest: /usr/local/lib/libpcl_stereo.so
StereoTest: /usr/local/lib/libpcl_kdtree.so
StereoTest: /usr/local/lib/libpcl_search.so
StereoTest: /usr/local/lib/libpcl_sample_consensus.so
StereoTest: /usr/local/lib/libpcl_filters.so
StereoTest: /usr/local/lib/libpcl_features.so
StereoTest: /usr/local/lib/libpcl_registration.so
StereoTest: /usr/local/lib/libpcl_ml.so
StereoTest: /usr/local/lib/libpcl_segmentation.so
StereoTest: /usr/local/lib/libpcl_visualization.so
StereoTest: /usr/local/lib/libpcl_surface.so
StereoTest: /usr/local/lib/libpcl_keypoints.so
StereoTest: /usr/local/lib/libpcl_tracking.so
StereoTest: /usr/local/lib/libpcl_recognition.so
StereoTest: /usr/local/lib/libpcl_people.so
StereoTest: /usr/local/lib/libpcl_outofcore.so
StereoTest: /usr/local/cuda/lib64/libcudart.so
StereoTest: /usr/local/cuda/lib64/libcurand.so
StereoTest: /usr/local/lib/libopencv_calib3d.so.3.4.3
StereoTest: /usr/local/lib/libopencv_features2d.so.3.4.3
StereoTest: /usr/local/lib/libopencv_highgui.so.3.4.3
StereoTest: /usr/local/lib/libopencv_flann.so.3.4.3
StereoTest: /usr/local/lib/libopencv_videoio.so.3.4.3
StereoTest: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
StereoTest: /usr/local/lib/libopencv_video.so.3.4.3
StereoTest: /usr/local/lib/libopencv_imgproc.so.3.4.3
StereoTest: /usr/local/lib/libopencv_core.so.3.4.3
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
StereoTest: /usr/lib/x86_64-linux-gnu/libtheoradec.so
StereoTest: /usr/lib/x86_64-linux-gnu/libogg.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkGeovisCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkproj4-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libxml2.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
StereoTest: /usr/lib/x86_64-linux-gnu/libsz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libdl.so
StereoTest: /usr/lib/x86_64-linux-gnu/libm.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
StereoTest: /usr/lib/x86_64-linux-gnu/libsz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libdl.so
StereoTest: /usr/lib/x86_64-linux-gnu/libm.so
StereoTest: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
StereoTest: /usr/lib/openmpi/lib/libmpi.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOExodus-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
StereoTest: /usr/lib/x86_64-linux-gnu/libnetcdf.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOLSDyna-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkverdict-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOSQL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkWrappingPython27Core-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkPythonInterpreter-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libpython2.7.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersAMR-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkParallelCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersFlowPaths-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOXML-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkViewsQt-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkViewsInfovis-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInfovisLayout-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersImaging-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libfreetype.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libz.so
StereoTest: /usr/lib/x86_64-linux-gnu/libGLU.so
StereoTest: /usr/lib/x86_64-linux-gnu/libGL.so
StereoTest: /usr/lib/x86_64-linux-gnu/libSM.so
StereoTest: /usr/lib/x86_64-linux-gnu/libICE.so
StereoTest: /usr/lib/x86_64-linux-gnu/libX11.so
StereoTest: /usr/lib/x86_64-linux-gnu/libXext.so
StereoTest: /usr/lib/x86_64-linux-gnu/libXt.so
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libvtksys-6.2.so.6.2.0
StereoTest: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.5.1
StereoTest: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.5.1
StereoTest: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.5.1
StereoTest: CMakeFiles/StereoTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/media/lietang/phenobot data/FLIR_stereo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable StereoTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/StereoTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/StereoTest.dir/build: StereoTest

.PHONY : CMakeFiles/StereoTest.dir/build

CMakeFiles/StereoTest.dir/requires: CMakeFiles/StereoTest.dir/src/stereo_opencv.cpp.o.requires
CMakeFiles/StereoTest.dir/requires: CMakeFiles/StereoTest.dir/StereoMatching.cpp.o.requires
CMakeFiles/StereoTest.dir/requires: CMakeFiles/StereoTest.dir/pm.cpp.o.requires

.PHONY : CMakeFiles/StereoTest.dir/requires

CMakeFiles/StereoTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/StereoTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/StereoTest.dir/clean

CMakeFiles/StereoTest.dir/depend:
	cd "/media/lietang/phenobot data/FLIR_stereo/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/media/lietang/phenobot data/FLIR_stereo" "/media/lietang/phenobot data/FLIR_stereo" "/media/lietang/phenobot data/FLIR_stereo/build" "/media/lietang/phenobot data/FLIR_stereo/build" "/media/lietang/phenobot data/FLIR_stereo/build/CMakeFiles/StereoTest.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/StereoTest.dir/depend

