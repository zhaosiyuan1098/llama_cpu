# CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 1. 项目定义
project(quick_infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 2. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# 3. 添加头文件目录
include_directories(
        include
        include/modules
        include/operations
        params
        kernel
)

# 4. 收集源文件
file(GLOB APP_SOURCES
        "src/main.cpp"
        "src/utlis.cpp"
        "src/llamaTokenlizer.cpp"
        "src/generate.cpp"
        "src/llamaGenerate.cpp"
)
file(GLOB MODULE_SOURCES "src/modules/*.cpp")
file(GLOB OPERATION_SOURCES "src/operations/*.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cpp")

# 5. 定义可执行文件
add_executable(quick_infer
        ${APP_SOURCES}
        ${MODULE_SOURCES}
        ${OPERATION_SOURCES}
        ${KERNEL_SOURCES}
)

target_compile_definitions(quick_infer PRIVATE
        "PROJECT_ROOT_DIR=\"${PROJECT_SOURCE_DIR}\""
)

# 6. 通过 IMP 宏设置 Matmul 实现

#--linear层乘法实现方式
#-- 0 最原始的实现
#-- 1 循环展开
#-- 2 多线程
#-- 3 simd
#-- 4 使用多线程+simd+循环展开加速
set(QUICK_INFER_MATMUL_IMPLEMENTATION "4" CACHE STRING "选择要使用的 matmul 实现 (0-4)")
set_property(CACHE QUICK_INFER_MATMUL_IMPLEMENTATION PROPERTY STRINGS "0" "1" "2" "3" "4")
target_compile_definitions(quick_infer PRIVATE "IMP=${QUICK_INFER_MATMUL_IMPLEMENTATION}")
# 通过 NUM_THREAD 宏设置使用线程数量
set(NUM_THREADS "12" CACHE STRING "Set the number of threads for multithreading")
target_compile_definitions(quick_infer PRIVATE "NUM_THREAD_MATMUL=${NUM_THREADS}")

# --- 从这里开始修复 ---

# 7. 平台相关的编译器选项 (已修正)
# 这个代码块现在复现了 xmake.lua 中的逻辑
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386")
    message(STATUS "检测到 x86 架构，添加 AVX2/FMA 标志。")
    target_compile_definitions(quick_infer PRIVATE QM_x86)
    target_compile_options(quick_infer PRIVATE -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    message(STATUS "检测到 ARM 架构，添加 NEON 标志。")
    target_compile_definitions(quick_infer PRIVATE QM_ARM)
    target_compile_options(quick_infer PRIVATE -mfpu=neon)
endif()

# 8. 链接必要的库 (已添加)
target_link_libraries(quick_infer PRIVATE pthread)

# --- 修复结束 ---

message(STATUS "quick_infer 配置成功！")
message(STATUS "Matmul 实现方式 (通过 IMP 宏选择): ${QUICK_INFER_MATMUL_IMPLEMENTATION}")