cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(CheckCXXCompilerFlag)

if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

project(cubemapgen VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

find_package(turbojpeg REQUIRED)
if (turbojpeg_FOUND)
  include_directories(${turbojpeg_INCLUDE_DIR})
endif()

find_package(PNG REQUIRED)
if (PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
endif()

find_package(OpenEXR 2.2.0 REQUIRED)
if (OpenEXR_FOUND)
  include_directories("${OPENEXR_INCLUDE_DIR}" "${ILMBASE_INCLUDE_DIR}" "${ILMBASE_INCLUDE_DIR}/OpenEXR")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node" LINKER_LANGUAGE CXX MACOSX_RPATH TRUE)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} ${turbojpeg_LIBRARY} ${PNG_LIBRARY} ${OPENEXR_LIBRARIES})
