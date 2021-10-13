
#include <stdio.h>

#include "cuda_draw.h"

namespace cudraw {

/******************************************************************************
 Internal Functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 ******************************************************************************/

namespace {

// return distance
// type 1: point outside segment near first point
// type 2: point outside segment near second point
// type 3: point inside the segment
__device__ float minDistanceToLineSegment(
  int x0, int y0, int x1, int y1, int x, int y, int& type) {
  // vector AB
  float AB_x = x1 - x0;
  float AB_y = y1 - y0;

  // vector BE
  float BE_x = x - x1;
  float BE_y = y - y1;

  // vector AE
  float AE_x = x - x0;
  float AE_y = y - y0;

  // Calculating the dot product
  float AB_BE = AB_x * BE_x + AB_y * BE_y;
  float AB_AE = AB_x * AE_x + AB_y * AE_y;

  // Minimum distance from
  // point E to the line segment
  float res = 0;
  type = 0;

  if (AB_BE > 0) {
    // Finding the magnitude
    res = sqrtf((x - x1) * (x - x1) + (y - y1) * (y - y1));
    type = 1;
  } else if (AB_AE < 0) {
    res = sqrtf((x - x0) * (x - x0) + (y - y0) * (y - y0));
    type = 2;
  } else {
    // Finding the perpendicular distance
    float mod = sqrt(AB_x * AB_x + AB_y * AB_y);
    res = abs(AB_x * AE_y - AB_y * AE_x) / mod;
    type = 3;
  }

  return res;
}

// Thread per block: 1D - 512
// Blocks per grid: 2D - height, (width + 511) / 512
__global__ void checkPixel(
    size_t width, size_t height, uint8_t* d_img, Line line, AABB aabb) {
  size_t i = blockIdx.x;
  size_t j = blockIdx.y * 512 + threadIdx.x;

  if (i < height && j < width && aabb.inside(j, i)) {
    size_t index_single = i * width + j;
    Color prev_color(d_img[3 * index_single], 
                     d_img[3 * index_single + 1], 
                     d_img[3 * index_single + 2]);
    auto color = line.checkPixel(j, i, prev_color);
    d_img[3 * index_single] = color.r;
    d_img[3 * index_single + 1] = color.g;
    d_img[3 * index_single + 2] = color.b;
  }
}

}

/******************************************************************************
 Internal Functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 ******************************************************************************/

/******************************************************************************
 Utilities >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 ******************************************************************************/

uint8_t* allocateImage(size_t width, size_t height) {
  uint8_t* d_res = nullptr;
  cudaMalloc(&d_res, width * height * 3);
  return d_res;  
}

uint8_t* uploadImage(size_t width, size_t height, uint8_t* img) {
  uint8_t* d_res = allocateImage(width, height);
  CHECK(d_res);
  cudaMemcpy(d_res, img, width * height * 3, cudaMemcpyHostToDevice);
  return d_res;  
}

void uploadImage(size_t width, size_t height, uint8_t* img, uint8_t* d_img) {
  CHECK(img);
  CHECK(d_img);
  cudaMemcpy(d_img, img, width * height * 3, cudaMemcpyHostToDevice);
}

void downloadImage(size_t width, size_t height, uint8_t* d_img, uint8_t* img) {
  cudaMemcpy(img, d_img, width * height * 3, cudaMemcpyDeviceToHost);
}

void copyImage(uint8_t* d_dst, uint8_t* d_src, size_t size) {
  cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
}

void freeImage(uint8_t* d_img) {
  cudaFree(d_img);
}

/******************************************************************************
 Utilities <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 ******************************************************************************/

/******************************************************************************
 API >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 ******************************************************************************/

__device__ Color Line::checkPixel(int x, int y, Color prev_color) {
  int type = 0;
  float dist = minDistanceToLineSegment(x0, y0, x1, y1, x, y, type);

  if (type == 3 && dist <= width / 2.f) {
    float frac = dist - floorf(dist);
    if (dist >= width / 2.f - 1.f) {
      Color c = prev_color;
      c.overlay(color, frac);
      return c;
    } else {
      return color;
    }
  } else {
    // TODO: add cap types
    return prev_color;
  }
}

__device__ __host__ AABB Line::getAABB() const {
  AABB res(x0 < x1 ? x0 : x1, 
           y0 < y1 ? y0 : y1, 
           x0 < x1 ? x1 : x0, 
           y0 < y1 ? y1 : y0);
  res.inflate(width);
  return res; 
}

void scanPixel(size_t width, size_t height, uint8_t* d_img, const Line& line) {
  CHECK(d_img);
  AABB aabb = line.getAABB();
  dim3 gridDims(height, (width + 511) / 512);
  checkPixel<<<gridDims, 512>>>(width, height, d_img, line, aabb);
}

/******************************************************************************
 API <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 ******************************************************************************/

}
