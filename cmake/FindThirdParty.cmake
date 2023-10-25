find_package(Eigen3 REQUIRED)
find_package(Assimp REQUIRED)

if( POLICY CMP0072 )
  cmake_policy(SET CMP0072 NEW)
endif()

find_package(OpenGL QUIET REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)


