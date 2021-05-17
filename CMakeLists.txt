cmake_minimum_required(VERSION 3.4)
project (interop)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O0")

if(NOT DEFINED ENV{DLDT_DIR})
    message(FATAL_ERROR "ERROR: env variable DLDT_DIR is not set")
endif()

if(NOT DEFINED ENV{DLDT_INCLUDE_DIR})
    message(FATAL_ERROR "ERROR: env variable DLDT_INCLUDE_DIR is not set")
endif()

if (DEFINED ENV{DLDT_DNNL})
    if($ENV{DLDT_DNNL} STREQUAL "on")
        message(STATUS "USE oneDNN plugin")
        add_definitions( -DDLDT_DNNL )
        set(GPUPluginLib DNNLPlugin)
    else()
        message(STATUS "USE clDNN plugin")
        set(GPUPluginLib clDNNPlugin)
    endif()
endif()

set (DLDT_DIR $ENV{DLDT_DIR})
set (DLDT_INCLUDE_DIR $ENV{DLDT_INCLUDE_DIR})
message("INFO: DLDT_DIR=" ${DLDT_DIR})
message("INFO: DLDT_INCLUDE_DIR=" ${DLDT_INCLUDE_DIR})

set (OpenCV_INCLUDE_DIR ${DLDT_DIR}/opencv/include)
set (OpenCV_LIB_DIR ${DLDT_DIR}/opencv/lib)
set (OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_photo opencv_highgui opencv_imgcodecs)
set (OPENVINO_IE_INCLUDE ${DLDT_INCLUDE_DIR}/include)
set (OPENVINO_IE_LIB_DIR ${DLDT_DIR}/lib)
set (OPENVINO_IE_LIBS ${GPUPluginLib} inference_engine inference_engine_legacy)

include_directories( ${OpenCV_INCLUDE_DIR} ${OPENVINO_IE_INCLUDE})
link_directories(${OpenCV_LIB_DIR} ${OPENVINO_IE_LIB_DIR})

add_executable(interop ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

set_target_properties(interop PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(interop PRIVATE ${OpenCV_LIBS} ${OPENVINO_IE_LIBS} OpenCL va va-drm)
