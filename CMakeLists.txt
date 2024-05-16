cmake_minimum_required(VERSION 3.28.5)
project(VulkanProject VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Bring in libraries.
include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.3.266
)

FetchContent_MakeAvailable(fmt)
FetchContent_MakeAvailable(glm)
FetchContent_MakeAvailable(glfw)
FetchContent_MakeAvailable(stb)
FetchContent_MakeAvailable(VulkanHeaders)

if (NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "VULKAN_SDK environment variable not set! Download Vulkan SDK v1.3.266")
endif()

set(VULKAN_SDK "$ENV{VULKAN_SDK}")

link_directories("${VULKAN_SDK}/Lib")
find_package(Vulkan REQUIRED)

include(CTest)
enable_testing()

add_executable(VulkanProject "source/main.cpp")
set_target_properties(VulkanProject PROPERTIES CXX_STANDARD 23)

target_compile_features(VulkanProject PRIVATE cxx_std_23)

# Enable SSE4 instructions and disable exceptions.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(VulkanProject PRIVATE -msse4.1 -msse4.2 -fno-exceptions)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(VulkanProject PRIVATE /arch:SSE4.2 /EHs-c-)
else()
    message(FATAL_ERROR "Unknown compiler. SSE4 not enabled!")
endif()

target_include_directories(VulkanProject PRIVATE ${stb_SOURCE_DIR})

# Link Vulkan
target_include_directories(VulkanProject PRIVATE "${VulkanHeaders_SOURCE_DIR}/include")
target_link_libraries(VulkanProject
    PRIVATE fmt::fmt
    PRIVATE glm::glm
    PRIVATE glfw
    PRIVATE Vulkan::Vulkan
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
