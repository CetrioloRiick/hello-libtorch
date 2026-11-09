cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hello_libtorch LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH "/opt/libtorch" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cxxopts)

add_executable(main
    src/main.cpp
    src/martino.cpp
)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}" cxxopts::cxxopts)
set_property(TARGET main PROPERTY CXX_STANDARD ${TORCH_CXX_STANDARD})

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:main>
    )
endif()
