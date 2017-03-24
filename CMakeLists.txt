cmake_minimum_required( VERSION 3.5 )

project( IntracranialPressure CXX )

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
else()
  find_package(ItkVtkGlue REQUIRED)
  include(${ItkVtkGlue_USE_FILE})
  set(Glue ItkVtkGlue)
endif()

set( source_files
  src/Main.cpp
  )

set( include_files
  include/IterationCallback.hpp
  )

add_executable( IntracranialPressure
  ${source_files}
  ${include_files}
  )

target_link_libraries( IntracranialPressure
  ${Glue}
  ${VTK_LIBRARIES}
  ${ITK_LIBRARIES}
  )