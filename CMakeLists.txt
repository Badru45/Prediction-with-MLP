# CMakeList.txt : CMake project for OpenCL-pro, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

#Adding Opencl libs and include files to the proj
find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})


find_package(Boost 1.56 REQUIRED)

#adding the Boost libs to the proj
if (WIN32)
  set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
  set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

  include_directories(${BOOST_INC})
  link_directories(${BOOST_LIB})

elseif(UNIX)
  message("OS: Linux")
  include_directories(${Boost_INCLUDE_DIR})
endif()


#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "Core/*.cpp" "Core/*.c")
file(GLOB OPENCL_SRC "OpenCL/*.cpp" "OpenCL/*.c")

if (WIN32)
  file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
endif()

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add source to this project's executable.
add_executable (OpenCL-pro "src/pro.cpp" ${CORE_SRC} ${OPENCL_SRC} )
target_include_directories (OpenCL-pro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src")
target_link_libraries (OpenCL-pro ${OpenCL_LIBRARY} dl boost_system) #imagehlp)

# TODO: Add tests and install targets if needed.
