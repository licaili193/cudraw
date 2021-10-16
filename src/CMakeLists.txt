find_package(CUDA REQUIRED)

cuda_add_library (cuda_draw SHARED 
  cuda_draw.cu
)

target_include_directories (cuda_draw PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries (cuda_draw ${CUDA_LINK_LIBRARIES_KEYWORD} 
  ${CUDA_LIBRARY}
)


cuda_add_executable (cuda_draw_example 
    cuda_draw_example.cu
)

target_link_libraries (cuda_draw_example ${CUDA_LINK_LIBRARIES_KEYWORD} 
    cuda
    glog
    cuda_draw
    ${OpenCV_LIBS}
    ${CUDA_LIBRARY}
    ${GFLAGS_LIBRARIES}
)

cuda_add_executable (cuda_draw_example_2 
    cuda_draw_example_2.cu
)

target_link_libraries (cuda_draw_example_2 ${CUDA_LINK_LIBRARIES_KEYWORD} 
    cuda
    glog
    cuda_draw
    ${OpenCV_LIBS}
    ${CUDA_LIBRARY}
    ${GFLAGS_LIBRARIES}
)
