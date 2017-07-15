cmake_minimum_required(VERSION 3.7)
project(ray_tracing)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)
if (OpenGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES})
endif()

find_package(GLU REQUIRED)
if (GLU_FOUND)
    include_directories(${GLU_INCLUDE_DIRS})
    link_libraries(${GLU_LIBRARIES})
endif()

find_package(GLUT REQUIRED)
if (GLUT_FOUND)
    include_directories(${GLUT_INCLUDE_DIRS})
    link_libraries(${GLUT_LIBRARIES})
endif()

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

set(SOURCE_FILES main.cpp)
add_executable(ray_tracing ${SOURCE_FILES})