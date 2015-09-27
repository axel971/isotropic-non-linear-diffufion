CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT(exec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# librairie OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )



# Chercher les fichiers sources
FILE(
    GLOB_RECURSE
    FICHIERS_SOURCES
    source/*
)


ADD_EXECUTABLE(${PROJECT_NAME} ${FICHIERS_SOURCES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})



