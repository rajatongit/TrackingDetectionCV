cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project( TDCV456 )
find_package( OpenCV REQUIRED )
# OpenCV
#FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS
#    core highgui imgproc video
#    REQUIRED
#)
#IF(OPENCV_CORE_FOUND)
#    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
#ENDIF()
#TARGET_LINK_LIBRARIES(object_detection ${OpenCV_LIBS})
add_executable( object_detection object_detection.cpp )

target_link_libraries( object_detection ${OpenCV_LIBS} )