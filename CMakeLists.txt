cmake_minimum_required(VERSION 3.16)
project(opencltest)  # 将 MyProject 替换为你的项目名称

set(include_paths
    /usr/local/cuda-11.8/targets/x86_64-linux/include
 
    )
 
set(link_paths
    /usr/local/cuda-11.8/targets/x86_64-linux/lib
)

add_executable(opencltest
  add_number/main.cpp
)
 
target_include_directories(opencltest PRIVATE
    ${include_paths}
)
find_package(OpenCL REQUIRED)
target_link_libraries(opencltest OpenCL::OpenCL)
target_link_directories(opencltest PRIVATE
    ${link_paths}
)
