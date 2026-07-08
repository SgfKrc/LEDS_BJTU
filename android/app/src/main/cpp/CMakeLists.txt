cmake_minimum_required(VERSION 3.22.1)

project(qlh_llama_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ANDROID_ABI)
    message(STATUS "QLH Android ABI: ${ANDROID_ABI}")
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "Unsupported ABI for local GGUF inference: ${ANDROID_ABI}")
    endif()
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libs" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Do not build llama common utils" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama app" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Disable host-native flags for Android cross compile" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Avoid libomp packaging for the first Android build" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Avoid build-time KleidiAI download" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Use statically linked backend" FORCE)

add_subdirectory(llama.cpp build-llama)

add_library(qlh_llama_jni SHARED qlh_llama_jni.cpp)

target_include_directories(qlh_llama_jni PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_LIST_DIR}/llama.cpp/ggml/include
)

target_link_libraries(qlh_llama_jni
    llama
    android
    log
)
