cmake_minimum_required(VERSION 3.10)
project(franka_real_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka 0.9.2 REQUIRED)

add_executable(go_home go_home.cpp)
target_link_libraries(go_home Franka::Franka)
install(TARGETS go_home DESTINATION bin)
