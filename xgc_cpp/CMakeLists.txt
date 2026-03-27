cmake_minimum_required(VERSION 3.14)
project(xgc_surrogate_infer CXX)

set(CMAKE_CXX_STANDARD 14)

option(USE_ONNXRUNTIME "Build with ONNX Runtime" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install root (e.g. /usr or conda env)")

if(USE_ONNXRUNTIME)
  if(ONNXRUNTIME_ROOT)
    set(onnxruntime_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
      set(onnxruntime_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    elseif(EXISTS "${ONNXRUNTIME_ROOT}/lib64/libonnxruntime.so")
      set(onnxruntime_LIBRARIES "${ONNXRUNTIME_ROOT}/lib64/libonnxruntime.so")
    else()
      set(onnxruntime_LIBRARIES "onnxruntime")
    endif()
  else()
    # Try system install or pkg-config
    find_path(onnxruntime_INCLUDE_DIRS onnxruntime_cxx_api.h
      HINTS /usr/include /usr/local/include)
    find_library(onnxruntime_LIBRARIES onnxruntime
      HINTS /usr/lib /usr/lib64 /usr/local/lib)
  endif()
endif()

add_library(xgc_surrogate_infer STATIC
  xgc_surrogate_infer.cpp
)

target_include_directories(xgc_surrogate_infer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_ONNXRUNTIME AND onnxruntime_INCLUDE_DIRS)
  target_include_directories(xgc_surrogate_infer PRIVATE ${onnxruntime_INCLUDE_DIRS})
  target_link_libraries(xgc_surrogate_infer PRIVATE ${onnxruntime_LIBRARIES})
  target_compile_definitions(xgc_surrogate_infer PRIVATE USE_ONNXRUNTIME)
else()
  message(STATUS "ONNX Runtime not found; building stub (inference will return -1)")
endif()

# Optional: standalone test executable (only when ONNX Runtime available)
if(USE_ONNXRUNTIME AND onnxruntime_LIBRARIES)
  add_executable(xgc_surrogate_test main_test.cpp)
  target_link_libraries(xgc_surrogate_test PRIVATE xgc_surrogate_infer ${onnxruntime_LIBRARIES})
endif()
